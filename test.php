<?php

//$code = <<<EOT
//typedef int32_t                     __LAPACK_int;
//
//void cblas_saxpy(const __LAPACK_int N, const float ALPHA, const float * X,
//                 const __LAPACK_int INCX, float *  Y, const __LAPACK_int INCY);
//EOT;
//$code = file_get_contents(__DIR__.'/src/cblas_new_macos.h');
//$code = file_get_contents(__DIR__.'/src/openblas.h');


//$filename = '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib';
//$filename = 'libopenblas.dll';
//
//
//$ffi = FFI::cdef($code,$filename);
//if($ffi==false) {
//    echo "LOAD ERROR\n";
//    return;
//}
//$n = 3;
//$x = $ffi->new("float[$n]");
//$y = $ffi->new("float[$n]");
//$x[0] = 1;
//$x[1] = 2;
//$x[2] = 3;
//$y[0] = 30;
//$y[1] = 20;
//$y[2] = 10;
//$ffi->cblas_saxpy($n,1.0,$x,1,$y,1);
//foreach($y as $v) {
//    var_dump($v);
//}

$loader = include __DIR__.'/vendor/autoload.php';
use Interop\Polite\Math\Matrix\NDArray;
$factory = new Rindow\OpenBLAS\FFI\OpenBLASFactory();
$buffer = new Rindow\Math\Buffer\FFI\BufferFactory();
$blas = $factory->Blas();
$n = 3;
$x = $buffer->Buffer($n,NDArray::float32);
$y = $buffer->Buffer($n,NDArray::float32);
$x[0] = 1;
$x[1] = 2;
$x[2] = 3;
$y[0] = 30;
$y[1] = 20;
$y[2] = 10;
$blas->axpy($n,1.0,$x,0,1,$y,0,1);
for($i=0;$i<$n;$i++) {
    var_dump($y[$i]);
}

$lapack = $factory->Lapackb();
echo "lapackb loaded\n";

/*
$loader = include __DIR__.'/vendor/autoload.php';
$loader->addPsr4('Rindow\\Math\\Matrix\\',__DIR__.'/../rindow-math-matrix/src');
$loader->addPsr4('Rindow\\Math\\Matrix\\Drivers\\MatlibFFI\\',__DIR__.'/../rindow-math-matrix-matlibffi/src');
$loader->addPsr4('Rindow\\Matlib\\FFI\\',__DIR__.'/../rindow-matlib-ffi/src');
$factory = new Rindow\OpenBLAS\FFI\OpenBLASFactory();
$mo = new Rindow\Math\Matrix\MatrixOperator(verbose:10);
echo $mo->service()->info();

$lapack = $factory->lapackb();
$matrix = $mo->array([
    [ 8.79,  9.93,  9.83,  5.45,  3.16,],
    [ 6.11,  6.91,  5.04, -0.27,  7.98,],
    [-9.15, -7.93,  4.86,  4.85,  3.01,],
    [ 9.57,  1.64,  8.83,  0.74,  5.80,],
    [-3.49,  4.02,  9.80, 10.00,  4.27,],
    [ 9.84,  0.15, -8.99, -6.02, -5.31,],
]);
$fullMatrices ??=true;
[$m,$n] = $matrix->shape();
if($fullMatrices) {
    $jobu  = ord('A');
    $jobvt = ord('A');
    $ldA = $n;
    $ldU = $m;
    $ldVT = $n;
} else {
    $jobu  = ord('S');
    $jobvt = ord('S');
    $ldA = $n;
    $ldU = min($m,$n);
    #$ldVT = min($m,$n);
    $ldVT = $n; // bug in the lapacke ???
}
$S = $mo->zeros([max($m,$n)],$matrix->dtype());
$U = $mo->zeros([$m,$ldU],$matrix->dtype());
$VT = $mo->zeros([$ldVT,$n],$matrix->dtype());
$SuperB = $mo->zeros([max($m,$n)-1],$matrix->dtype());

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

const LAPACK_ROW_MAJOR = 101;
$matrix_layout = LAPACK_ROW_MAJOR;
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
echo "S=".$mo->toString($S,'%6.2f',indent:true)."\n";
echo "U=".$mo->toString($U,'%6.2f',indent:true)."\n";
echo "VT=".$mo->toString($VT,'%6.2f',indent:true)."\n";
*/
