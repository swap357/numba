"""Platform-wise signext/zeroext (issue #10802)."""

import re
import unittest

import llvmlite.binding as ll

from numba import cfunc, jit, types
from numba.core.cpu import argext
from numba.tests.support import TestCase


_COLS = (
    (8, True), (8, False), (16, True), (16, False),
    (32, True), (32, False), (64, True), (64, False),
)

# gist.github.com/sklam/17dd8d44294045e3644b43bcf9b7060a
_PARAM = {
    'aarch64-unknown-linux-gnu': (None,) * 8,
    'armv7-unknown-linux-gnueabihf': (
        'signext', 'zeroext', 'signext', 'zeroext', None, None, None, None,
    ),
    'i386-pc-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext', None, None, None, None,
    ),
    'loongarch64-unknown-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext',
        'signext', 'signext', None, None,
    ),
    'mips64-unknown-linux-gnuabi64': (
        'signext', 'zeroext', 'signext', 'zeroext',
        'signext', 'signext', 'signext', 'zeroext',
    ),
    'powerpc-unknown-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext', None, None, None, None,
    ),
    'powerpc64-unknown-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext',
        'signext', 'zeroext', None, None,
    ),
    'powerpc64le-unknown-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext',
        'signext', 'zeroext', None, None,
    ),
    'riscv64-unknown-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext',
        'signext', 'signext', None, None,
    ),
    's390x-ibm-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext',
        'signext', 'zeroext', None, None,
    ),
    'x86_64-pc-linux-gnu': (
        'signext', 'zeroext', 'signext', 'zeroext', None, None, None, None,
    ),
    'x86_64-pc-windows-msvc': (None,) * 8,
}


class TestArgext(TestCase):

    def test_parameters(self):
        for triple, row in _PARAM.items():
            for (width, signed), expected in zip(_COLS, row):
                got = argext(triple, width, signed, is_param=True)
                self.assertEqual(got, expected, msg=(triple, width, signed))

    def test_returns(self):
        # Same as params except mips64 i64/u64
        for triple, row in _PARAM.items():
            for (width, signed), expected in zip(_COLS, row):
                if triple.startswith('mips64') and width == 64:
                    expected = None
                got = argext(triple, width, signed, is_param=False)
                self.assertEqual(got, expected, msg=(triple, width, signed))

    def test_unknown_triple(self):
        self.assertIsNone(argext('wasm32-unknown-unknown', 8, True))


class TestCEdges(TestCase):

    def test_cfunc_wrapper(self):
        triple = ll.get_process_triple()

        @cfunc(types.int8(types.int8), nopython=True)
        def ident(a):
            return a

        llvm_ir = ident.inspect_llvm()
        match = re.search(r'define[^\n]*@cfunc\.[^(]+\(([^)]*)\)', llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        expected = argext(triple, 8, True, True)
        if expected:
            self.assertIn(expected, match.group(1))
        else:
            self.assertNotIn('signext', match.group(1))
            self.assertNotIn('zeroext', match.group(1))

    def test_jit_internal_not_decorated(self):
        if argext(ll.get_process_triple(), 8, True) is None:
            self.skipTest('host C ABI does not extend i8')

        @jit(types.int8(types.int8), nopython=True)
        def ident(x):
            return x

        llvm_ir = ident.inspect_llvm(ident.signatures[0])
        for ln in llvm_ir.splitlines():
            if ln.startswith('define') and '@cfunc.' not in ln:
                self.assertNotIn('signext', ln)
                self.assertNotIn('zeroext', ln)


if __name__ == '__main__':
    unittest.main()
