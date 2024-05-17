<?php
namespace RindowTest\OpenBLAS\FFI\LapackeTest;

require_once __DIR__.'/LapackbTest.php';
use RindowTest\OpenBLAS\FFI\LapackbTest\LapackbTest;

/**
 * @requires OS WINNT|Linux
 */
class LapackeTest extends LapackbTest
{
    public function getLapack()
    {
        $lapack = $this->factory->Lapack();
        return $lapack;
    }
}
