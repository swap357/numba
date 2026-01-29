"""Unit tests for IR parsing."""
import subprocess
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    parse_numba_ir,
    parse_bytecode,
    parse_type_inference,
    parse_llvm_ir,
    parse_passes,
    extract_main_function_ir,
    extract_main_function_llvm,
    run_with_env,
)

TEST_CODE = """from numba import njit

@njit
def f():
    x = None
    for _ in range(0):
        x = 'v'
    return x == None

f()
"""

SIMPLE_CODE = """from numba import njit

@njit
def add(a, b):
    return a + b

add(1, 2)
"""


def test_parse_numba_ir_simple():
    """Test parsing a simple IR block."""
    ir_text = """IR DUMP: f
label 0:
    x = const(int, 42)                       ['x']
    return x                                 ['x']
"""
    result = parse_numba_ir(ir_text)
    assert result['function'] == 'f', f"Expected 'f', got {result['function']}"
    assert len(result['blocks']) == 1, f"Expected 1 block, got {len(result['blocks'])}"
    assert result['blocks'][0]['label'] == 0
    assert len(result['blocks'][0]['instructions']) == 2
    print("✓ test_parse_numba_ir_simple")


def test_parse_numba_ir_with_jumps():
    """Test parsing IR with control flow."""
    ir_text = """IR DUMP: loop
label 0:
    i = const(int, 0)                        ['i']
    jump 10                                  []
label 10:
    cond = i < 10                            ['cond', 'i']
    branch cond, 20, 30                      ['cond']
label 20:
    i = i + 1                                ['i']
    jump 10                                  []
label 30:
    return i                                 ['i']
"""
    result = parse_numba_ir(ir_text)
    assert result['function'] == 'loop'
    assert len(result['blocks']) == 4

    # Check successors
    block0 = next(b for b in result['blocks'] if b['label'] == 0)
    assert block0['successors'] == [10], f"Block 0 successors: {block0['successors']}"

    block10 = next(b for b in result['blocks'] if b['label'] == 10)
    assert set(block10['successors']) == {20, 30}, f"Block 10 successors: {block10['successors']}"

    block30 = next(b for b in result['blocks'] if b['label'] == 30)
    assert block30['successors'] == [], f"Block 30 should have no successors"

    print("✓ test_parse_numba_ir_with_jumps")


def test_parse_bytecode():
    """Test bytecode parsing."""
    bc_text = """>          0	NOP(arg=None, lineno=3)
           2	RESUME(arg=0, lineno=3)
           4	LOAD_CONST(arg=0, lineno=5)
           6	STORE_FAST(arg=0, lineno=5)
"""
    result = parse_bytecode(bc_text)
    assert len(result) == 4, f"Expected 4 instructions, got {len(result)}"
    assert result[0]['offset'] == 0
    assert result[0]['opcode'] == 'NOP'
    assert result[2]['lineno'] == 5
    print("✓ test_parse_bytecode")


def test_parse_type_inference():
    """Test type inference parsing."""
    ti_text = """---------------------------------Variable types---------------------------------
{'x': none,
 'y': int64,
 '$result': bool}
----------------------------------Return type-----------------------------------
bool
-----------------------------------Call types-----------------------------------
{foo(x): (int64,) -> bool}
"""
    result = parse_type_inference(ti_text)
    assert result['return_type'] == 'bool', f"Return type: {result['return_type']}"
    assert 'x' in result['variables']
    assert result['variables']['y'] == 'int64'
    print("✓ test_parse_type_inference")


def test_extract_main_function_ir():
    """Test extracting main function from multi-function IR dump."""
    raw = """-----------------------------------IR DUMP: f-----------------------------------
label 0:
    x = const(int, 1)                        ['x']
    jump 10                                  []
label 10:
    return x                                 ['x']

------------IR DUMP: gen_non_eq.<locals>.none_equality.<locals>.impl------------
label 0:
    a = arg(0, name=a)                       ['a']
    return a                                 ['a']
"""
    result = extract_main_function_ir(raw, raw)
    assert result['function'] == 'f', f"Expected 'f', got {result['function']}"
    assert len(result['blocks']) == 2, f"Expected 2 blocks, got {len(result['blocks'])}"
    print("✓ test_extract_main_function_ir")


def test_real_numba_ir_output():
    """Test with actual numba output."""
    stdout, stderr, code = run_with_env(SIMPLE_CODE, "NUMBA_DUMP_IR")
    raw = stderr + stdout

    assert 'IR DUMP' in raw, f"No IR DUMP in output. stderr: {stderr[:200]}"

    result = extract_main_function_ir(raw, raw)
    assert result['function'] == 'add', f"Expected 'add', got {result['function']}"
    assert len(result['blocks']) >= 1, f"Expected at least 1 block, got {len(result['blocks'])}"

    # Verify block structure
    for block in result['blocks']:
        assert 'label' in block
        assert 'instructions' in block
        assert 'successors' in block
        assert isinstance(block['instructions'], list)

    print(f"✓ test_real_numba_ir_output (found {len(result['blocks'])} blocks)")


def test_real_numba_typeinfer():
    """Test with actual type inference output."""
    stdout, stderr, code = run_with_env(SIMPLE_CODE, "NUMBA_DEBUG_TYPEINFER")
    raw = stderr + stdout

    assert 'Variable types' in raw or 'type variables' in raw, "No type info in output"

    result = parse_type_inference(raw)
    assert result['return_type'] is not None, "No return type found"
    assert len(result['variables']) > 0, "No variables found"

    print(f"✓ test_real_numba_typeinfer (return: {result['return_type']}, vars: {len(result['variables'])})")


def test_parse_passes():
    """Test parsing pass-by-pass IR output."""
    stdout, stderr, code = run_with_env(SIMPLE_CODE, "NUMBA_DEBUG_PRINT_AFTER", value="all")
    raw = stderr + stdout

    assert 'AFTER' in raw, "No AFTER markers in output"

    passes = parse_passes(raw)
    assert len(passes) > 0, "No passes found"

    # Check pass structure
    for p in passes[:3]:
        assert 'pass_name' in p, "Missing pass_name"
        assert 'ir' in p, "Missing ir"
        assert 'blocks' in p['ir'], "Missing blocks in ir"

    # Filter to main function
    main_passes = [p for p in passes if '__main__' in p['function']]
    assert len(main_passes) > 5, f"Expected many passes, got {len(main_passes)}"

    # Check some known passes exist
    pass_names = [p['pass_name'] for p in main_passes]
    assert any('translate_bytecode' in pn for pn in pass_names), "Missing translate_bytecode pass"

    print(f"✓ test_parse_passes (found {len(main_passes)} passes for main function)")


def test_real_complex_ir():
    """Test with more complex code that has control flow."""
    stdout, stderr, code = run_with_env(TEST_CODE, "NUMBA_DUMP_IR")
    raw = stderr + stdout

    result = extract_main_function_ir(raw, raw)

    assert result['function'] == 'f', f"Expected 'f', got {result['function']}"
    assert len(result['blocks']) >= 3, f"Expected at least 3 blocks (loop), got {len(result['blocks'])}"

    # Should have blocks with jumps/branches
    has_jump = any(b['successors'] for b in result['blocks'])
    assert has_jump, "No control flow edges found"

    print(f"✓ test_real_complex_ir (blocks: {len(result['blocks'])}, edges: {sum(len(b['successors']) for b in result['blocks'])})")


def test_api_response_structure():
    """Test the full API response structure."""
    import json
    import urllib.request

    try:
        req = urllib.request.Request(
            'http://127.0.0.1:5050/compile',
            data=json.dumps({'code': SIMPLE_CODE}).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"⚠ test_api_response_structure skipped (server not running): {e}")
        return

    # Check all stages present
    for stage in ['bytecode', 'ir', 'ssa', 'llvm', 'assembly']:
        assert stage in data, f"Missing stage: {stage}"
        assert 'parsed' in data[stage], f"Missing 'parsed' in {stage}"
        assert 'raw' in data[stage], f"Missing 'raw' in {stage}"

    # Check IR has blocks
    ir_parsed = data['ir']['parsed']
    assert ir_parsed is not None, "IR parsed is None"
    assert 'blocks' in ir_parsed, f"No 'blocks' in IR parsed: {ir_parsed.keys()}"
    assert len(ir_parsed['blocks']) > 0, "No blocks in IR"

    # Check SSA has both ir and types
    ssa_parsed = data['ssa']['parsed']
    assert 'ir' in ssa_parsed, "No 'ir' in SSA parsed"
    assert 'types' in ssa_parsed, "No 'types' in SSA parsed"
    assert len(ssa_parsed['ir']['blocks']) > 0, "No blocks in SSA IR"

    print(f"✓ test_api_response_structure (IR blocks: {len(ir_parsed['blocks'])}, SSA blocks: {len(ssa_parsed['ir']['blocks'])})")


if __name__ == '__main__':
    print("Running parser tests...\n")

    test_parse_numba_ir_simple()
    test_parse_numba_ir_with_jumps()
    test_parse_bytecode()
    test_parse_type_inference()
    test_extract_main_function_ir()
    test_real_numba_ir_output()
    test_real_numba_typeinfer()
    test_parse_passes()
    test_real_complex_ir()
    test_api_response_structure()

    print("\n✅ All tests passed!")
