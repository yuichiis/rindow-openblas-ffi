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

    public function getLapack()
    {
        $lapack = $this->factory->Lapackb();
        return $lapack;
    }

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

    public function translate_gesvd(
        NDArray $matrix,
        bool $fullMatrices=null,
        ) : array
    {
        if($matrix->ndim()!=2) {
            throw new InvalidArgumentException("input array must be 2D array");
        }
        $fullMatrices ??=true;
        [$m,$n] = $matrix->shape();
        if($fullMatrices) {
            $jobu  = 'A';
            $jobvt = 'A';
            $ldA = $n;
            $ldU = $m;
            $ldVT = $n;
        } else {
            $jobu  = 'S';
            $jobvt = 'S';
            $ldA = $n;
            $ldU = min($m,$n);
            #$ldVT = min($m,$n);
            $ldVT = $n; // bug in the lapacke ???
        }

        $S = $this->zeros([min($m,$n)],$matrix->dtype());
        $U = $this->zeros([$m,$ldU],$matrix->dtype());
        $VT = $this->zeros([$ldVT,$n],$matrix->dtype());
        $SuperB = $this->zeros([min($m,$n)-1],$matrix->dtype());

        $AA = $matrix->buffer();
        $offsetA = $matrix->offset();
        $SS = $S->buffer();
        $offsetS = $S->offset();
        $UU = $U->buffer();
        $offsetU = $U->offset();
        $VVT = $VT->buffer();
        $offsetVT = $VT->offset();
        $SuperBB = $SuperB->buffer();
        $offsetSuperB = $SuperB->offset();
        return [
            self::LAPACK_ROW_MAJOR,
            ord($jobu),
            ord($jobvt),
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $U,$S,$VT
        ];
        //if(!$fullMatrices) {
        //    // bug in the lapacke ???
        //    $VT = $this->copy($VT[[0,min($m,$n)-1]]);
        //}
        //return [$U,$S,$VT];
    }

    public function testSvdFull1()
    {
        $lapack = $this->getLapack();
        $a = $this->array([
            //[8.79,  6.11, -9.15,  9.57, -3.49,  9.84],
            //[9.93,  6.91, -7.93,  1.64,  4.02,  0.15],
            //[9.83,  5.04,  4.86,  8.83,  9.80, -8.99],
            //[5.45, -0.27,  4.85,  0.74, 10.00, -6.02],
            //[3.16,  7.98,  3.01,  5.80,  4.27, -5.31],
    

            [ 8.79,  9.93,  9.83,  5.45,  3.16,],
            [ 6.11,  6.91,  5.04, -0.27,  7.98,],
            [-9.15, -7.93,  4.86,  4.85,  3.01,],
            [ 9.57,  1.64,  8.83,  0.74,  5.80,],
            [-3.49,  4.02,  9.80, 10.00,  4.27,],
            [ 9.84,  0.15, -8.99, -6.02, -5.31,],
        ]);
        $fullMatrices = null;
        $this->assertEquals([6,5],$a->shape());
        [
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB,

            $u,$s,$vt
        ] = $this->translate_gesvd(
            $a,
            $fullMatrices,
        );

        $lapack->gesvd(
            $matrix_layout,
            $jobu,
            $jobvt,
            $m,
            $n,
            $AA,  $offsetA,  $ldA,
            $SS,  $offsetS,
            $UU,  $offsetU,  $ldU,
            $VVT, $offsetVT, $ldVT,
            $SuperBB,  $offsetSuperB
        );
        if(!$fullMatrices) {
            // bug in the lapacke ???
            //$vt = $this->copy($vt[[0,min($m,$n)-1]]);
        }
        // [$u,$s,$vt] = $la->svd($a);
        $this->assertEquals([6,6],$u->shape());
        $this->assertEquals([5],$s->shape());
        $this->assertEquals([5,5],$vt->shape());

        # echo "---- u ----\n";
        # foreach($u->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";
        # echo "---- s ----\n";
        # echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$s->toArray()))."],\n";
        # echo "---- vt ----\n";
        # foreach($vt->toArray() as $array)
        #     echo '['.implode(',',array_map(function($a){return sprintf('%5.2f',$a);},$array))."],\n";

        # ---- u ----
        $correctU = $this->array([
            [-0.59, 0.26, 0.36, 0.31, 0.23, 0.55],
            [-0.40, 0.24,-0.22,-0.75,-0.36, 0.18],
            [-0.03,-0.60,-0.45, 0.23,-0.31, 0.54],
            [-0.43, 0.24,-0.69, 0.33, 0.16,-0.39],
            [-0.47,-0.35, 0.39, 0.16,-0.52,-0.46],
            [ 0.29, 0.58,-0.02, 0.38,-0.65, 0.11],
        ]);
        //$this->assertTrue(false);
        //echo $this->arrayToString($u,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($u,$correctU,rtol:1e-2,atol:1e-3));
        #$this->assertLessThan(0.01,abs($this->amax($this->axpy($u,$correctU,-1))));
        # ---- s ----
        $correctS = $this->array(
            [27.47,22.64, 8.56, 5.99, 2.01]
        );
        //echo $this->arrayToString($s,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($s,$correctS,rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($s,$correctS,-1))));
        # ---- vt ----
        $correctVT = $this->array([
            [-0.25,-0.40,-0.69,-0.37,-0.41],
            [ 0.81, 0.36,-0.25,-0.37,-0.10],
            [-0.26, 0.70,-0.22, 0.39,-0.49],
            [ 0.40,-0.45, 0.25, 0.43,-0.62],
            [-0.22, 0.14, 0.59,-0.63,-0.44],
        ]);
        //echo $this->arrayToString($vt,'%10.6f',true)."\n";
        $this->assertTrue($this->isclose($vt,$correctVT,rtol:1e-2,atol:1e-3));
        //$this->assertLessThan(0.01,abs($this->amax($this->axpy($vt,$correctVT,-1))));
    }

}