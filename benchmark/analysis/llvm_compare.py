"""
LLVM IR and Assembly analysis for Numba functions.

Detects vectorization, SIMD usage, and optimization patterns.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum


class VectorWidth(Enum):
    """SIMD vector widths."""
    SCALAR = 1
    SSE = 128      # 128-bit (SSE)
    AVX = 256      # 256-bit (AVX/AVX2)
    AVX512 = 512   # 512-bit (AVX-512)


@dataclass
class LoopInfo:
    """Information about a loop in LLVM IR."""
    has_vector_ops: bool = False
    vector_width: int = 1
    estimated_iterations: Optional[int] = None
    has_unroll: bool = False
    has_reduction: bool = False
    reduction_type: str = ""  # "fadd", "fmul", "add", etc.


@dataclass
class SIMDInfo:
    """SIMD instruction usage information."""
    has_sse: bool = False
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_fma: bool = False
    vector_width: int = 1
    instructions: Dict[str, int] = field(default_factory=dict)


@dataclass
class LLVMAnalysis:
    """Complete LLVM IR and assembly analysis result."""
    # Vectorization
    is_vectorized: bool = False
    vector_width: int = 1
    simd_info: SIMDInfo = field(default_factory=SIMDInfo)

    # Loop analysis
    loop_count: int = 0
    loops: List[LoopInfo] = field(default_factory=list)

    # Instruction counts
    total_instructions: int = 0
    memory_ops: int = 0
    arithmetic_ops: int = 0
    branch_ops: int = 0
    call_ops: int = 0

    # Patterns detected
    has_early_exit: bool = False
    has_function_calls: bool = False
    called_functions: List[str] = field(default_factory=list)
    has_indirect_calls: bool = False

    # LLVM optimization hints
    optimization_remarks: List[str] = field(default_factory=list)

    # Raw IR snippets for debugging
    ir_summary: str = ""
    asm_summary: str = ""


# Patterns for LLVM IR analysis
_VECTOR_TYPE_PATTERN = re.compile(r'<(\d+)\s*x\s*(i\d+|float|double)>')
_LOOP_PATTERN = re.compile(r'(\.loop\d*|\.for\.|\.while\.)')
_REDUCTION_PATTERN = re.compile(r'(fadd|fmul|add|mul)\s+.*\s+(fast|reassoc)')
_CALL_PATTERN = re.compile(r'call\s+[^@]*@([^\(]+)')
_BRANCH_PATTERN = re.compile(r'\s+br\s+')
_LOAD_PATTERN = re.compile(r'\s+load\s+')
_STORE_PATTERN = re.compile(r'\s+store\s+')

# Patterns for x86 assembly analysis
_SSE_REGISTERS = re.compile(r'\b(xmm\d+)\b')
_AVX_REGISTERS = re.compile(r'\b(ymm\d+)\b')
_AVX512_REGISTERS = re.compile(r'\b(zmm\d+)\b')
_SIMD_INSTRUCTIONS = {
    'sse': re.compile(r'\b(addps|addss|subps|subss|mulps|mulss|divps|divss|'
                      r'addpd|addsd|subpd|subsd|mulpd|mulsd|divpd|divsd|'
                      r'movaps|movups|movapd|movupd|sqrtps|sqrtss|sqrtpd|sqrtsd)\b'),
    'avx': re.compile(r'\b(vaddps|vaddss|vsubps|vsubss|vmulps|vmulss|vdivps|vdivss|'
                      r'vaddpd|vaddsd|vsubpd|vsubsd|vmulpd|vmulsd|vdivpd|vdivsd|'
                      r'vmovaps|vmovups|vmovapd|vmovupd|vsqrtps|vsqrtss|vsqrtpd|vsqrtsd)\b'),
    'fma': re.compile(r'\b(vfmadd|vfmsub|vfnmadd|vfnmsub)\d*[ps][sd]\b'),
    'avx512': re.compile(r'\b(vaddps|vsubps|vmulps|vdivps)\b.*\bzmm'),
}


def _analyze_llvm_ir(ir: str) -> Dict[str, Any]:
    """Analyze LLVM IR for optimization patterns."""
    result = {
        'is_vectorized': False,
        'vector_width': 1,
        'loop_count': 0,
        'has_reduction': False,
        'reduction_type': '',
        'called_functions': [],
        'has_early_exit': False,
        'memory_ops': 0,
        'arithmetic_ops': 0,
        'branch_ops': 0,
        'call_ops': 0,
    }

    # Check for vector types
    vector_matches = _VECTOR_TYPE_PATTERN.findall(ir)
    if vector_matches:
        result['is_vectorized'] = True
        widths = [int(m[0]) for m in vector_matches]
        result['vector_width'] = max(widths) if widths else 1

    # Count loops
    result['loop_count'] = len(_LOOP_PATTERN.findall(ir))

    # Check for reductions
    reduction_matches = _REDUCTION_PATTERN.findall(ir)
    if reduction_matches:
        result['has_reduction'] = True
        result['reduction_type'] = reduction_matches[0][0]

    # Find called functions
    calls = _CALL_PATTERN.findall(ir)
    # Filter out LLVM intrinsics we don't care about
    external_calls = [c for c in calls if not c.startswith('llvm.')]
    result['called_functions'] = list(set(external_calls))
    result['call_ops'] = len(calls)

    # Check for early exit (conditional return in loop)
    if 'ret ' in ir and result['loop_count'] > 0:
        # Heuristic: if there's a return inside loop-like structure
        lines = ir.split('\n')
        in_loop = False
        for line in lines:
            if _LOOP_PATTERN.search(line):
                in_loop = True
            if in_loop and 'ret ' in line:
                result['has_early_exit'] = True
                break

    # Count instruction types
    result['memory_ops'] = len(_LOAD_PATTERN.findall(ir)) + len(_STORE_PATTERN.findall(ir))
    result['branch_ops'] = len(_BRANCH_PATTERN.findall(ir))

    # Count arithmetic (rough estimate)
    arith_ops = ['fadd', 'fsub', 'fmul', 'fdiv', 'add ', 'sub ', 'mul ', 'sdiv', 'udiv']
    for op in arith_ops:
        result['arithmetic_ops'] += ir.count(op)

    return result


def _analyze_assembly(asm: str) -> SIMDInfo:
    """Analyze x86 assembly for SIMD usage."""
    info = SIMDInfo()

    # Check for SIMD registers
    if _SSE_REGISTERS.search(asm):
        info.has_sse = True
        info.vector_width = max(info.vector_width, 128)

    if _AVX_REGISTERS.search(asm):
        info.has_avx = True
        info.vector_width = max(info.vector_width, 256)

    if _AVX512_REGISTERS.search(asm):
        info.has_avx512 = True
        info.vector_width = max(info.vector_width, 512)

    # Check for FMA
    if _SIMD_INSTRUCTIONS['fma'].search(asm):
        info.has_fma = True
        info.has_avx2 = True  # FMA implies AVX2

    # Count SIMD instructions
    for category, pattern in _SIMD_INSTRUCTIONS.items():
        matches = pattern.findall(asm)
        if matches:
            info.instructions[category] = len(matches)

    return info


def analyze_numba_function(func: Callable, signature: tuple = None) -> LLVMAnalysis:
    """
    Analyze a Numba JIT-compiled function.

    Args:
        func: Numba JIT-compiled function
        signature: Optional signature tuple (e.g., (numba.float64[:],))

    Returns:
        LLVMAnalysis with detailed analysis results
    """
    analysis = LLVMAnalysis()

    try:
        # Get LLVM IR
        if signature:
            ir = func.inspect_llvm(signature)
        else:
            # Try to get from first available signature
            sigs = list(func.signatures)
            if sigs:
                ir = func.inspect_llvm(sigs[0])
            else:
                return analysis
    except (AttributeError, KeyError):
        # Not a Numba function or no compilation yet
        return analysis

    # Analyze IR
    ir_analysis = _analyze_llvm_ir(ir)
    analysis.is_vectorized = ir_analysis['is_vectorized']
    analysis.vector_width = ir_analysis['vector_width']
    analysis.loop_count = ir_analysis['loop_count']
    analysis.has_early_exit = ir_analysis['has_early_exit']
    analysis.called_functions = ir_analysis['called_functions']
    analysis.has_function_calls = len(ir_analysis['called_functions']) > 0
    analysis.memory_ops = ir_analysis['memory_ops']
    analysis.arithmetic_ops = ir_analysis['arithmetic_ops']
    analysis.branch_ops = ir_analysis['branch_ops']
    analysis.call_ops = ir_analysis['call_ops']

    # Create IR summary (first 500 chars for debugging)
    analysis.ir_summary = ir[:500] + "..." if len(ir) > 500 else ir

    # Try to get assembly
    try:
        if signature:
            asm = func.inspect_asm(signature)
        else:
            sigs = list(func.signatures)
            if sigs:
                asm = func.inspect_asm(sigs[0])
            else:
                asm = ""

        if asm:
            analysis.simd_info = _analyze_assembly(asm)
            # Update vector width from actual SIMD usage
            if analysis.simd_info.vector_width > analysis.vector_width:
                analysis.vector_width = analysis.simd_info.vector_width
                analysis.is_vectorized = analysis.vector_width > 1

            # Count total instructions (approximate)
            analysis.total_instructions = len([l for l in asm.split('\n')
                                               if l.strip() and not l.strip().startswith('.')])

            # Assembly summary
            analysis.asm_summary = asm[:500] + "..." if len(asm) > 500 else asm

    except (AttributeError, KeyError):
        pass

    return analysis


def detect_vectorization(func: Callable, signature: tuple = None) -> Tuple[bool, int, str]:
    """
    Quick check for vectorization in a Numba function.

    Args:
        func: Numba JIT-compiled function
        signature: Optional signature

    Returns:
        Tuple of (is_vectorized, vector_width, description)
    """
    analysis = analyze_numba_function(func, signature)

    if not analysis.is_vectorized:
        return False, 1, "Scalar code (no vectorization detected)"

    width = analysis.vector_width
    simd = analysis.simd_info

    if simd.has_avx512:
        desc = f"AVX-512 ({width} elements)"
    elif simd.has_avx or simd.has_avx2:
        fma_str = " with FMA" if simd.has_fma else ""
        desc = f"AVX/AVX2{fma_str} ({width} elements)"
    elif simd.has_sse:
        desc = f"SSE ({width} elements)"
    else:
        desc = f"Vectorized ({width} elements)"

    return True, width, desc


def compare_llvm_ir(func_a: Callable, func_b: Callable,
                    signature: tuple = None) -> Dict[str, Any]:
    """
    Compare LLVM analysis between two Numba functions.

    Args:
        func_a: First Numba function
        func_b: Second Numba function
        signature: Optional shared signature

    Returns:
        Dictionary with comparison results
    """
    analysis_a = analyze_numba_function(func_a, signature)
    analysis_b = analyze_numba_function(func_b, signature)

    return {
        'vectorization': {
            'a': (analysis_a.is_vectorized, analysis_a.vector_width),
            'b': (analysis_b.is_vectorized, analysis_b.vector_width),
            'diff': analysis_b.vector_width - analysis_a.vector_width,
        },
        'instruction_counts': {
            'a': analysis_a.total_instructions,
            'b': analysis_b.total_instructions,
            'diff': analysis_b.total_instructions - analysis_a.total_instructions,
        },
        'memory_ops': {
            'a': analysis_a.memory_ops,
            'b': analysis_b.memory_ops,
            'diff': analysis_b.memory_ops - analysis_a.memory_ops,
        },
        'has_early_exit': {
            'a': analysis_a.has_early_exit,
            'b': analysis_b.has_early_exit,
        },
        'function_calls': {
            'a': analysis_a.called_functions,
            'b': analysis_b.called_functions,
        },
        'analysis_a': analysis_a,
        'analysis_b': analysis_b,
    }


def format_analysis(analysis: LLVMAnalysis) -> str:
    """Format LLVMAnalysis as human-readable string."""
    lines = [
        "LLVM Analysis Results",
        "=" * 40,
        "",
        f"Vectorization: {'Yes' if analysis.is_vectorized else 'No'}",
    ]

    if analysis.is_vectorized:
        lines.append(f"  Vector width: {analysis.vector_width} elements")
        simd = analysis.simd_info
        if simd.has_avx512:
            lines.append("  SIMD: AVX-512")
        elif simd.has_avx2:
            lines.append("  SIMD: AVX2" + (" + FMA" if simd.has_fma else ""))
        elif simd.has_avx:
            lines.append("  SIMD: AVX")
        elif simd.has_sse:
            lines.append("  SIMD: SSE")

    lines.extend([
        "",
        f"Loops: {analysis.loop_count}",
        f"Early exit: {'Yes' if analysis.has_early_exit else 'No'}",
        "",
        "Instruction counts:",
        f"  Total: ~{analysis.total_instructions}",
        f"  Memory ops: {analysis.memory_ops}",
        f"  Arithmetic ops: {analysis.arithmetic_ops}",
        f"  Branch ops: {analysis.branch_ops}",
        f"  Call ops: {analysis.call_ops}",
    ])

    if analysis.called_functions:
        lines.append("")
        lines.append("External calls:")
        for fn in analysis.called_functions[:5]:  # Limit to 5
            lines.append(f"  - {fn}")
        if len(analysis.called_functions) > 5:
            lines.append(f"  ... and {len(analysis.called_functions) - 5} more")

    return "\n".join(lines)
