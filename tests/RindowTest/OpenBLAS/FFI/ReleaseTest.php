<?php
namespace RindowTest\OpenBLAS\FFI\ReleaseTest;

use PHPUnit\Framework\TestCase;
use FFI;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use Rindow\OpenBLAS\FFI\Blas;
use Rindow\OpenBLAS\FFI\Lapack;

class ReleaseTest extends TestCase
{
    public function testLoadClasses()
    {
        $ffi = FFI::cdef('');
        $factory = new OpenBLASFactory();
        $blas = new Blas($ffi);
        $lapack = new Lapack($ffi);
        $this->assertTrue(true);
    }
}