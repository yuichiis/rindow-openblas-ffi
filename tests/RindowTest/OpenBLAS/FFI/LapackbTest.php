<?php
namespace RindowTest\OpenBLAS\FFI\LapackbTest;

require_once __DIR__.'/LapackTest.php';
use RindowTest\OpenBLAS\FFI\LapackTest\LapackTest;

class LapackbTest extends LapackTest
{
    public function getLapack()
    {
        $lapack = $this->factory->Lapackb();
        return $lapack;
    }
}
