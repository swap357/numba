"""Numba compilation pipeline visualizer with graph rendering."""
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Numba Pipeline Viewer")

STAGES = [
    ("bytecode", "NUMBA_DUMP_BYTECODE", "Bytecode"),
    ("ir", "NUMBA_DUMP_IR", "Numba IR"),
    ("passes", "NUMBA_DEBUG_PRINT_AFTER", "IR Passes"),
    ("ssa", "NUMBA_DEBUG_TYPEINFER", "SSA + Types"),
    ("llvm", "NUMBA_DUMP_LLVM", "LLVM IR"),
    ("assembly", "NUMBA_DUMP_ASSEMBLY", "Assembly"),
]


def run_with_env(code: str, env_var: str, value: str = "1") -> tuple[str, str, int]:
    """Run code with a specific NUMBA env var enabled."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            env = os.environ.copy()
            env[env_var] = value
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            return result.stdout, result.stderr, result.returncode
        finally:
            os.unlink(f.name)


@dataclass
class IRBlock:
    label: int
    instructions: list[str]
    successors: list[int]  # jump/branch targets


@dataclass
class TypeInfo:
    name: str
    type_str: str


def parse_numba_ir(output: str) -> dict:
    """Parse Numba IR into blocks for CFG visualization."""
    blocks = []
    current_block = None
    func_name = None

    for line in output.split('\n'):
        # Stop if we hit a new function/pass header (---...AFTER...---)
        if re.match(r'^-{5,}.*:\s*\w+:\s*AFTER\s+', line):
            break

        # Match function header
        if 'IR DUMP' in line:
            match = re.search(r'IR DUMP[:\s]+(\w+)', line)
            if match:
                func_name = match.group(1)
            continue

        # Match block label
        label_match = re.match(r'^label (\d+):', line)
        if label_match:
            if current_block:
                blocks.append(current_block)
            current_block = {
                'label': int(label_match.group(1)),
                'instructions': [],
                'successors': [],
            }
            continue

        if current_block is None:
            continue

        line = line.strip()
        if not line or line.startswith('['):
            continue

        # Parse instructions
        current_block['instructions'].append(line)

        # Extract control flow (strip trailing comment before parsing)
        code_part = line.split('[')[0].strip() if '[' in line else line
        if code_part.startswith('jump '):
            match = re.search(r'jump (\d+)', code_part)
            if match:
                current_block['successors'].append(int(match.group(1)))
        elif code_part.startswith('branch '):
            # branch $cond, true_label, false_label
            # Extract the two numeric labels after the condition
            match = re.search(r'branch\s+[^,]+,\s*(\d+),\s*(\d+)', code_part)
            if match:
                current_block['successors'].extend([int(match.group(1)), int(match.group(2))])
        elif code_part.startswith('return '):
            pass  # No successors

    if current_block:
        blocks.append(current_block)

    return {'function': func_name or 'unknown', 'blocks': blocks}


def parse_type_inference(output: str) -> dict:
    """Parse type inference output into structured data."""
    variables = {}
    return_type = None
    call_types = []

    # Parse variable types section
    var_section = re.search(r"Variable types.*?\n\{([^}]+)\}", output, re.DOTALL)
    if var_section:
        for match in re.finditer(r"'([^']+)':\s*([^,}]+)", var_section.group(1)):
            var_name = match.group(1)
            var_type = match.group(2).strip()
            variables[var_name] = var_type

    # Parse return type
    ret_match = re.search(r"Return type.*?\n([^\n]+)", output)
    if ret_match:
        return_type = ret_match.group(1).strip('-').strip()

    # Parse call types
    call_section = re.search(r"Call types.*?\n\{([^}]*)\}", output, re.DOTALL)
    if call_section:
        for match in re.finditer(r"([^:]+):\s*\(([^)]*)\)\s*->\s*([^,}]+)", call_section.group(1)):
            call_types.append({
                'call': match.group(1).strip(),
                'args': match.group(2).strip(),
                'result': match.group(3).strip(),
            })

    return {
        'variables': variables,
        'return_type': return_type,
        'call_types': call_types,
    }


def parse_llvm_ir(output: str) -> dict:
    """Parse LLVM IR into basic blocks."""
    blocks = []
    current_block = None
    func_name = None

    for line in output.split('\n'):
        # Match function definition
        func_match = re.match(r'define\s+\S+\s+@"?([^"(]+)', line)
        if func_match:
            func_name = func_match.group(1)
            continue

        # Match basic block label
        block_match = re.match(r'^(\w+):', line)
        if block_match and not line.startswith('define'):
            if current_block:
                blocks.append(current_block)
            current_block = {
                'label': block_match.group(1),
                'instructions': [],
                'successors': [],
            }
            continue

        if current_block is None:
            continue

        line = line.strip()
        if not line or line.startswith(';'):
            continue

        current_block['instructions'].append(line)

        # Extract branches
        if 'br ' in line:
            # br i1 %cond, label %then, label %else
            # br label %next
            labels = re.findall(r'label %"?(\w+)"?', line)
            current_block['successors'].extend(labels)
        elif 'ret ' in line:
            pass  # No successors

    if current_block:
        blocks.append(current_block)

    return {'function': func_name or 'unknown', 'blocks': blocks}


def analyze_llvm_ir(raw: str) -> dict:
    """Count exact LLVM IR opcodes."""
    lines = raw.split('\n')

    opcodes = {}  # opcode -> count
    vector_ops = []  # list of (opcode, width, element_type)
    calls = {}  # function name -> count
    blocks = 0
    phi_count = 0

    # Pattern to extract opcode from LLVM IR instruction
    # Handles: %x = fadd double %a, %b -> "fadd"
    #          store double %x, double* %p -> "store"
    #          br label %exit -> "br"
    #          ret i64 %result -> "ret"
    opcode_pattern = re.compile(
        r'(?:%[\w.]+\s*=\s*)?'  # optional assignment
        r'\b(add|sub|mul|[us]div|[us]rem|shl|lshr|ashr|and|or|xor|'
        r'fadd|fsub|fmul|fdiv|frem|fneg|'
        r'icmp|fcmp|'
        r'load|store|alloca|getelementptr|'
        r'br|switch|ret|invoke|resume|unreachable|phi|select|'
        r'call|'
        r'trunc|zext|sext|fptrunc|fpext|fptoui|fptosi|uitofp|sitofp|'
        r'ptrtoint|inttoptr|bitcast|addrspacecast|'
        r'extractelement|insertelement|extractvalue|insertvalue|shufflevector|'
        r'fence|cmpxchg|atomicrmw)\b'
    )
    vector_type_pattern = re.compile(r'<(\d+)\s*x\s*(\w+)>')
    call_func_pattern = re.compile(r'call[^@]*@"?([^"(\s]+)')

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue

        # Skip header lines (=== ... === or --- ... ---)
        if stripped.startswith('=') or stripped.startswith('-'):
            continue

        # Skip lines that look like dump headers (LLVM DUMP, etc.)
        if 'DUMP' in stripped:
            continue

        # Count block labels
        block_match = re.match(r'^(\w+):', stripped)
        if block_match:
            blocks += 1
            continue

        # Skip declarations and metadata
        if stripped.startswith('define') or stripped.startswith('declare'):
            continue
        if stripped.startswith('!') or stripped.startswith('target'):
            continue
        if stripped.startswith('attributes'):
            continue
        if stripped.startswith('}') or stripped.startswith('{'):
            continue

        # Extract opcode
        opcode_match = opcode_pattern.search(stripped)
        if opcode_match:
            opcode = opcode_match.group(1)
            opcodes[opcode] = opcodes.get(opcode, 0) + 1

            # Track phi nodes
            if opcode == 'phi':
                phi_count += 1

            # Track vector types
            vec_match = vector_type_pattern.search(stripped)
            if vec_match:
                width = vec_match.group(1)
                elem_type = vec_match.group(2)
                vector_ops.append((opcode, width, elem_type))

            # Track called functions
            if opcode == 'call':
                func_match = call_func_pattern.search(stripped)
                if func_match:
                    func_name = func_match.group(1)
                    calls[func_name] = calls.get(func_name, 0) + 1

    # Dedupe vector types
    vector_types = list(set(f"<{w} x {t}>" for _, w, t in vector_ops))

    return {
        'total': sum(opcodes.values()),
        'blocks': blocks,
        'opcodes': opcodes,
        'vector_ops': len(vector_ops),
        'vector_types': vector_types,
        'calls': calls,
        'phi': phi_count,
    }


def analyze_assembly(raw: str) -> dict:
    """Count exact assembly mnemonics."""
    lines = raw.split('\n')

    mnemonics = {}  # mnemonic -> count
    labels = 0
    registers = {'xmm': 0, 'ymm': 0, 'zmm': 0}
    memory_ops = {'load': 0, 'store': 0}

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#') or stripped.startswith(';'):
            continue

        # Skip header lines (=== ... === or --- ... ---)
        if stripped.startswith('=') or stripped.startswith('-'):
            continue

        # Skip assembler directives (start with .)
        if stripped.startswith('.'):
            continue

        # Handle labels - anything ending with : that might have instruction after
        if ':' in stripped:
            # Find the last colon (handles addresses like "0x123: mov rax, rbx")
            colon_idx = stripped.rfind(':')
            after_colon = stripped[colon_idx + 1:].strip()
            if after_colon:
                # There's an instruction after the label/address
                stripped = after_colon
            else:
                # Just a label, no instruction
                labels += 1
                continue

        # Skip lines that start with quotes (quoted labels in some formats)
        if stripped.startswith('"') or stripped.startswith("'"):
            continue

        # Extract mnemonic - first alphabetic word
        # Split on whitespace and take first token
        parts = stripped.split()
        if not parts:
            continue

        first_word = parts[0]

        # Skip if it looks like a hex address
        if first_word.startswith('0x') or first_word.startswith('0X'):
            continue

        # Skip if it ends with : (label that slipped through)
        if first_word.endswith(':'):
            labels += 1
            continue

        # Skip if it starts with . (directive that slipped through)
        if first_word.startswith('.'):
            continue

        # The mnemonic - lowercase for consistency
        mnemonic = first_word.lower()
        mnemonics[mnemonic] = mnemonics.get(mnemonic, 0) + 1

        # Count register usage
        line_lower = stripped.lower()
        # Use regex to count distinct register references
        registers['xmm'] += len(re.findall(r'\bxmm\d*\b', line_lower))
        registers['ymm'] += len(re.findall(r'\bymm\d*\b', line_lower))
        registers['zmm'] += len(re.findall(r'\bzmm\d*\b', line_lower))

        # Heuristic for memory ops: instructions with [] (x86) or () (ARM)
        has_mem_ref = '[' in stripped or ('(' in stripped and not stripped.startswith('('))
        if has_mem_ref:
            # mov/ldr/str family - check destination
            if mnemonic.startswith(('mov', 'vmov', 'ldr', 'str', 'ld', 'st')):
                # On x86: first operand is destination
                # On ARM: for ldr dest is first, for str src is first
                if mnemonic.startswith(('str', 'st')):
                    memory_ops['store'] += 1
                elif mnemonic.startswith(('ldr', 'ld')):
                    memory_ops['load'] += 1
                else:
                    # mov - check if dest has memory reference
                    comma_idx = stripped.find(',')
                    if comma_idx > 0:
                        dest_part = stripped[:comma_idx]
                        if '[' in dest_part or '(' in dest_part:
                            memory_ops['store'] += 1
                        else:
                            memory_ops['load'] += 1
                    else:
                        memory_ops['load'] += 1
            elif mnemonic in ('push', 'call', 'stp'):
                memory_ops['store'] += 1
            elif mnemonic in ('pop', 'ret', 'ldp'):
                memory_ops['load'] += 1
            else:
                memory_ops['load'] += 1

    return {
        'total': sum(mnemonics.values()),
        'labels': labels,
        'mnemonics': mnemonics,
        'registers': registers,
        'memory_ops': memory_ops,
    }


def parse_bytecode(output: str) -> list[dict]:
    """Parse bytecode into structured instructions."""
    instructions = []
    for line in output.split('\n'):
        match = re.match(r'^[>\s]*(\d+)\s+(\w+)\(arg=([^,]+),\s*lineno=(\d+)\)', line)
        if match:
            instructions.append({
                'offset': int(match.group(1)),
                'opcode': match.group(2),
                'arg': match.group(3),
                'lineno': int(match.group(4)),
            })
    return instructions


def parse_passes(output: str, func_name: str = None) -> list[dict]:
    """Parse NUMBA_DEBUG_PRINT_AFTER output into list of passes."""
    passes = []

    # Pattern: ---func: mode: AFTER pass_name---
    # [^:\n]+ ensures we only match within the header line
    pattern = r'-{3,}([^:\n]+):\s*(\w+):\s*AFTER\s+([^-\n]+)-{3,}\n(.*?)(?=\n-{3,}[^:\n]+:\s*\w+:\s*AFTER|\Z)'
    matches = re.findall(pattern, output, re.DOTALL)

    for func, mode, pass_name, content in matches:
        func = func.strip()
        pass_name = pass_name.strip()

        # Filter to main function if specified
        if func_name and func_name not in func:
            continue

        # Parse the IR content
        ir = parse_numba_ir(f"IR DUMP: {func}\n{content}")

        passes.append({
            'function': func,
            'mode': mode,
            'pass_name': pass_name,
            'ir': ir,
            'raw': content.strip(),
        })

    return passes


def extract_main_function_ir(output: str, raw: str) -> dict:
    """Extract just the main user function from IR dumps."""
    # Find IR DUMP sections - pattern: ---IR DUMP: name---\n<content>
    # Content ends at next section header (10+ dashes followed by IR DUMP) or end
    pattern = r'-+\s*((?:SSA )?IR DUMP[:\s]+([^\s-]+))\s*-+\n(.*?)(?=\n-{10,}\s*(?:SSA )?IR DUMP|\Z)'
    matches = re.findall(pattern, raw, re.DOTALL)

    for header, func_name, content in matches:
        # Skip internal helpers (have 'locals' in the header)
        if 'locals' not in header.lower():
            return parse_numba_ir(f"{header}\n{content}")

    # Fallback: just parse the whole thing
    return parse_numba_ir(raw)


def extract_main_function_llvm(output: str) -> dict:
    """Extract just the main function from LLVM dumps."""
    # Find LLVM DUMP sections - pattern: ---LLVM DUMP <desc>---\n<content>\n===
    pattern = r'-+\s*(LLVM DUMP[^-]+)\s*-+\n(.*?)(?=\n={20,}|\Z)'
    matches = re.findall(pattern, output, re.DOTALL)

    for header, content in matches:
        # Prefer the main function
        if '__main__' in header and 'nrt' not in header.lower():
            return parse_llvm_ir(content)

    # Fallback
    for header, content in matches:
        if '__main__' in content:
            return parse_llvm_ir(content)

    return parse_llvm_ir(output)


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path(__file__).parent.joinpath("index.html").read_text()


@app.post("/compile")
async def compile_code(request: Request):
    data = await request.json()
    code = data.get('code', '')
    if not code.strip():
        return {"error": "No code provided"}

    results = {}
    for stage_id, env_var, label in STAGES:
        stdout, stderr, returncode = run_with_env(code, env_var)
        raw = stderr + stdout

        if stage_id == "bytecode":
            parsed = parse_bytecode(raw)
            results[stage_id] = {
                "label": label,
                "parsed": parsed,
                "raw": raw,
                "returncode": returncode,
            }
        elif stage_id == "ir":
            parsed = extract_main_function_ir(raw, raw)
            results[stage_id] = {
                "label": label,
                "parsed": parsed,
                "raw": raw,
                "returncode": returncode,
            }
        elif stage_id == "ssa":
            ir_parsed = extract_main_function_ir(raw, raw)
            type_parsed = parse_type_inference(raw)
            results[stage_id] = {
                "label": label,
                "parsed": {
                    "ir": ir_parsed,
                    "types": type_parsed,
                },
                "raw": raw,
                "returncode": returncode,
            }
        elif stage_id == "passes":
            # Use "all" to get all passes
            stdout, stderr, returncode = run_with_env(code, env_var, value="all")
            raw = stderr + stdout
            parsed = parse_passes(raw)
            # Extract unique functions (preserve order, main function first)
            functions = []
            seen = set()
            for p in parsed:
                if p['function'] not in seen:
                    functions.append(p['function'])
                    seen.add(p['function'])
            results[stage_id] = {
                "label": label,
                "parsed": parsed,
                "functions": functions,
                "raw": raw,
                "returncode": returncode,
            }
            continue
        elif stage_id == "llvm":
            parsed = extract_main_function_llvm(raw)
            analysis = analyze_llvm_ir(raw)
            results[stage_id] = {
                "label": label,
                "parsed": parsed,
                "analysis": analysis,
                "raw": raw,
                "returncode": returncode,
            }
        elif stage_id == "assembly":
            analysis = analyze_assembly(raw)
            results[stage_id] = {
                "label": label,
                "parsed": None,
                "analysis": analysis,
                "raw": raw,
                "returncode": returncode,
            }
        else:
            results[stage_id] = {
                "label": label,
                "parsed": None,
                "raw": raw,
                "returncode": returncode,
            }

    return results


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050)
