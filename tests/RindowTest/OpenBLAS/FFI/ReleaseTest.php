<?php
namespace RindowTest\OpenBLAS\FFI\ReleaseTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\OpenBLAS\FFI\OpenBLASFactory;
use Rindow\OpenBLAS\FFI\Blas;
use Rindow\OpenBLAS\FFI\Lapack;
use FFI;

require_once __DIR__.'/Utils.php';
use RindowTest\OpenBLAS\FFI\Utils;

class ReleaseTest extends TestCase
{
    use Utils;
    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected $factory;

    public function testFFINotLoaded()
    {
        $factory = $this->factory;
        if(extension_loaded('ffi')) {
            $blas = $factory->Blas();
            $lapack = $factory->Lapack();
            $this->assertInstanceof(Blas::class,$blas);
            $this->assertInstanceof(Lapack::class,$lapack);
        } else {
            $this->assertFalse($factory->isAvailable());
        }
    }
}