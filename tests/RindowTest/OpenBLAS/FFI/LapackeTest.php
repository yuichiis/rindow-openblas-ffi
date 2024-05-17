<?php
namespace RindowTest\OpenBLAS\FFI\LapackeTest;

require_once __DIR__.'/LapackTest.php';
use RindowTest\OpenBLAS\FFI\LapackTest\LapackTest;

/**
 * @requires OS WINNT|Linux
 */
class LapackeTest extends LapackTest
{
    public function getLapack()
    {
        $lapack = $this->factory->Lapack();
        return $lapack;
    }
}
