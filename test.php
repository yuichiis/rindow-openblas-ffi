<?php

$code = <<<EOT
typedef int32_t                     __LAPACK_int;

void cblas_saxpy(const __LAPACK_int N, const float ALPHA, const float * X,
                 const __LAPACK_int INCX, float *  Y, const __LAPACK_int INCY);
EOT;

$filename = '/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib';
//$filename = 'libopenblas.dll';

$ffi = FFI::cdef($code,$filename);
if($ffi==false) {
    echo "LOAD ERROR\n";
    return;
}
$n = 3;
$x = $ffi->new("float[$n]");
$y = $ffi->new("float[$n]");
$x[0] = 1;
$x[1] = 2;
$x[2] = 3;
$y[0] = 30;
$y[1] = 20;
$y[2] = 10;
$ffi->cblas_saxpy($n,1.0,$x,1,$y,1);
foreach($y as $v) {
    var_dump($v);
}
