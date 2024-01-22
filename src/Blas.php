<?php
namespace Rindow\OpenBLAS\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS as BLASIF;
use InvalidArgumentException;
use FFI;

use Interop\Polite\Math\Matrix\LinearBuffer as BufferInterface;

class Blas
{
    use Utils;

    // OpenBLAS is compiled for sequential use
    const OPENBLAS_SEQUENTIAL = 0;
    // OpenBLAS is compiled using normal threading model
    const OPENBLAS_THREAD = 1;
    // OpenBLAS is compiled using OpenMP threading model
    const OPENBLAS_OPENMP = 2;

    protected object $ffi;

    public function __construct(FFI $ffi)
    {
        $this->ffi = $ffi;
    }

    public function getNumThreads() : int
    {
        return $this->ffi->openblas_get_num_threads();
    }

    public function getNumProcs() : int
    {
        return $this->ffi->openblas_get_num_procs();
    }

    public function getConfig() : string
    {
        $string = $this->ffi->openblas_get_config();
        return FFI::string($string);
    }

    public function getCorename() : string
    {
        $string = $this->ffi->openblas_get_corename();
        return FFI::string($string);
    }

    public function getParallel() : int
    {
        return $this->ffi->openblas_get_parallel();
    }


    /**
     *  X := alpha * X
     */
    public function scal(
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX) : void
    {
        $ffi= $this->ffi;
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);
        switch($X->dtype()) {
            case NDArray::float32:{
                $ffi->cblas_sscal($n,$alpha,$X->addr($offsetX),$incX);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dscal($n,$alpha,$X->addr($offsetX),$incX);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    /**
     *  Y := alpha * X + Y
     */
    public function axpy(
        int $n,
        float $alpha,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y, $n, $offsetY, $incY);

        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $ffi->cblas_saxpy($n,$alpha,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_daxpy($n,$alpha,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function dot(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY ) : float
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y, $n, $offsetY, $incY);

        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $result = $ffi->cblas_sdot($n,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            case NDArray::float64:{
                $result = $ffi->cblas_ddot($n,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $result;
    }

    public function asum(
        int $n,
        BufferInterface $X, int $offsetX, int $incX ) : float
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);

        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);

        switch($X->dtype()) {
            case NDArray::float32:{
                $result = $ffi->cblas_sasum($n,$X->addr($offsetX),$incX);
                break;
            }
            case NDArray::float64:{
                $result = $ffi->cblas_dasum($n,$X->addr($offsetX),$incX);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $result;
    }

    public function iamax(
        int $n,
        BufferInterface $X, int $offsetX, int $incX ) : int
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);

        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);

        switch($X->dtype()) {
            case NDArray::float32:{
                $result = $ffi->cblas_isamax($n,$X->addr($offsetX),$incX);
                break;
            }
            case NDArray::float64:{
                $result = $ffi->cblas_idamax($n,$X->addr($offsetX),$incX);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $result;
    }

    public function iamin(
        int $n,
        BufferInterface $X, int $offsetX, int $incX ) : int
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);

        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);

        switch($X->dtype()) {
            case NDArray::float32:{
                $result = $ffi->cblas_isamin($n,$X->addr($offsetX),$incX);
                break;
            }
            case NDArray::float64:{
                $result = $ffi->cblas_idamin($n,$X->addr($offsetX),$incX);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $result;
    }

    public function copy(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y, $n, $offsetY, $incY);

        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $result = $ffi->cblas_scopy($n,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            case NDArray::float64:{
                $result = $ffi->cblas_dcopy($n,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            default: {
                if($incX==1&&$incY==1) {
                    $bytes = $n*$X->value_size();
                    FFI::memcpy($Y->addr($offsetY),$X->addr($offsetX),$bytes);
                } else {
                    $xx = $X->addr($offsetX);
                    $yy = $Y->addr($offsetY);
                    for($i=0,$idX=0,$idY=0; $i<$n; $i++,$idX+=$incX,$idY+=$incY) {
                        $yy[$idY] = $xx[$idX];
                    }
                }
            }
        }
    }

    public function nrm2(
        int $n,
        BufferInterface $X, int $offsetX, int $incX
        ) : float
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);

        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);

        switch($X->dtype()) {
            case NDArray::float32:{
                $result = $ffi->cblas_snrm2($n,$X->addr($offsetX),$incX);
                break;
            }
            case NDArray::float64:{
                $result = $ffi->cblas_dnrm2($n,$X->addr($offsetX),$incX);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $result;
    }

    public function rotg(
        BufferInterface $A, int $offsetA,
        BufferInterface $B, int $offsetB,
        BufferInterface $C, int $offsetC,
        BufferInterface $S, int $offsetS
        ) : void
    {
        $ffi= $this->ffi;

        // Check Buffer A
        $this->assert_vector_buffer_spec("A", $A, 1, $offsetA, 1);
        // Check Buffer B
        $this->assert_vector_buffer_spec("B", $B, 1, $offsetB, 1);
        // Check Buffer C
        $this->assert_vector_buffer_spec("C", $C, 1, $offsetC, 1);
        // Check Buffer S
        $this->assert_vector_buffer_spec("S", $S, 1, $offsetS, 1);

        // Check Buffer A and B and C and S
        $dtype = $A->dtype();
        if($dtype!=$B->dtype()||$dtype!=$C->dtype()||$dtype!=$S->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A,B,C and S");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_srotg(
                    $A->addr($offsetA),
                    $B->addr($offsetB),
                    $C->addr($offsetC),
                    $S->addr($offsetS),
                );
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_drotg(
                    $A->addr($offsetA),
                    $B->addr($offsetB),
                    $C->addr($offsetC),
                    $S->addr($offsetS),
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function rot(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY,
        BufferInterface $C, int $offsetC,
        BufferInterface $S, int $offsetS
        ) : void
    {
        $ffi= $this->ffi;

        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, 1, $offsetX, 1);
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y, 1, $offsetY, 1);
        // Check Buffer C
        $this->assert_vector_buffer_spec("C", $C, 1, $offsetC, 1);
        // Check Buffer S
        $this->assert_vector_buffer_spec("S", $S, 1, $offsetS, 1);

        // Check Buffer A and B and C and S
        $dtype = $X->dtype();
        if($dtype!=$Y->dtype()||$dtype!=$C->dtype()||$dtype!=$S->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A,B,C and S");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_srot(
                    $n,
                    $X->addr($offsetX),$incX,
                    $Y->addr($offsetY),$incY,
                    $C[$offsetC],
                    $S[$offsetS],
                );
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_drot(
                    $n,
                    $X->addr($offsetX),$incX,
                    $Y->addr($offsetY),$incY,
                    $C[$offsetC],
                    $S[$offsetS],
                );
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function swap(
        int $n,
        BufferInterface $X, int $offsetX, int $incX,
        BufferInterface $Y, int $offsetY, int $incY ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $n, $offsetX, $incX);
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y, $n, $offsetY, $incY);

        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }

        switch($X->dtype()) {
            case NDArray::float32:{
                $ffi->cblas_sswap($n,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_sswap($n,$X->addr($offsetX),$incX,$Y->addr($offsetY),$incY);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function gemv(
        int $order,
        int $trans,
        int $m,
        int $n,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $X, int $offsetX, int $incX,
        float $beta,
        BufferInterface $Y, int $offsetY, int $incY ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        // Check Buffer A
        $this->assert_matrix_buffer_spec("A", $A, $m, $n, $offsetA, $ldA);

        // Check Buffer size X and Y
        if($trans==BLASIF::NoTrans || $trans==BLASIF::ConjNoTrans ) {
            $rows = $m; $cols = $n;
        } elseif($trans==BLASIF::Trans || $trans==BLASIF::ConjTrans) {
            $rows = $n; $cols = $m;
        } else {
            throw new InvalidArgumentException("unknown transpose mode for bufferA.");
        }
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X, $cols, $offsetX, $incX);
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y, $rows, $offsetY, $incY);
    
        // Check Buffer A and X and Y
        $dtype = $A->dtype();
        if($dtype!=$X->dtype() || $dtype!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and X and Y");
        }
    
        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_sgemv(
                    $order, $trans,
                    $m, $n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $X->addr($offsetX),$incX,
                    $beta,
                    $Y->addr($offsetY),$incY);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dgemv(
                    $order, $trans,
                    $m, $n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $X->addr($offsetX),$incX,
                    $beta,
                    $Y->addr($offsetY),$incY);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function gemm(
        int $order,
        int $transA,
        int $transB,
        int $m,
        int $n,
        int $k,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $B, int $offsetB, int $ldB,
        float $beta,
        BufferInterface $C, int $offsetC, int $ldC ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);

        // Check Buffer A
        if($transA==BLASIF::NoTrans || $transA==BLASIF::ConjNoTrans) {
            $rows = $m; $cols = $k;
        } else if($transA==BLASIF::Trans || $transA==BLASIF::ConjTrans) {
            $rows = $k; $cols = $m;
        } else {
            throw new InvalidArgumentException('unknown transpose mode for bufferA.');
        }
        $this->assert_matrix_buffer_spec("A", $A, $rows, $cols, $offsetA, $ldA);

        // Check Buffer B
        if($transB==BLASIF::NoTrans || $transB==BLASIF::ConjNoTrans) {
            $rows = $k; $cols = $n;
        } elseif($transB==BLASIF::Trans || $transB==BLASIF::ConjTrans) {
            $rows = $n; $cols = $k;
        } else {
            throw new InvalidArgumentException('unknown transpose mode for bufferB.');
        }
        $this->assert_matrix_buffer_spec("B", $B, $rows, $cols, $offsetB, $ldB);

        // Check Buffer C
        $this->assert_matrix_buffer_spec("C", $C, $m, $n, $offsetC, $ldC);


        // Check Buffer A and B and C
        $dtype = $A->dtype();
        if($dtype!=$B->dtype() || $dtype!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B and C");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_sgemm(
                    $order,
                    $transA,
                    $transB,
                    $m,$n,$k,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dgemm(
                    $order,
                    $transA,
                    $transB,
                    $m,$n,$k,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function symm(
        int $order,
        int $side,
        int $uplo,
        int $m,
        int $n,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $B, int $offsetB, int $ldB,
        float $beta,
        BufferInterface $C, int $offsetC, int $ldC ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        // Check Buffer A
        if($side==BLASIF::Left) {
            $rows = $m;
        } elseif($side==BLASIF::Right) {
            $rows = $n;
        } else {
            throw new InvalidArgumentException('unknown side mode for bufferA.');
        }
        $this->assert_matrix_buffer_spec("A", $A, $rows, $rows, $offsetA, $ldA);

        // Check Buffer B
        $this->assert_matrix_buffer_spec("B", $B, $m, $n, $offsetB, $ldB);

        // Check Buffer C
        $this->assert_matrix_buffer_spec("C", $C, $m, $n, $offsetC, $ldC);

        // Check Buffer A and B and C
        $dtype = $A->dtype();
        if($dtype!=$B->dtype() || $dtype!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B and C");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_ssymm(
                    $order,
                    $side,
                    $uplo,
                    $m,$n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dsymm(
                    $order,
                    $side,
                    $uplo,
                    $m,$n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }

    }

    public function syrk(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        float $beta,
        BufferInterface $C, int $offsetC, int $ldC ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);

        if($trans==BLASIF::NoTrans || $trans==BLASIF::ConjNoTrans) {
            $rows = $n; $cols = $k;
        } else if($trans==BLASIF::Trans || $trans==BLASIF::ConjTrans) {
            $rows = $k; $cols = $n;
        } else {
            throw new InvalidArgumentException("unknown transpose mode for bufferA.");
        }
        $this->assert_matrix_buffer_spec("A", $A, $rows, $cols, $offsetA, $ldA);

        // Check Buffer C
        $this->assert_matrix_buffer_spec("C", $C, $n, $n, $offsetC, $ldC);

        // Check Buffer A and C
        $dtype = $A->dtype();
        if($dtype!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and C");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_ssyrk(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dsyrk(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function syr2k(
        int $order,
        int $uplo,
        int $trans,
        int $n,
        int $k,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $B, int $offsetB, int $ldB,
        float $beta,
        BufferInterface $C, int $offsetC, int $ldC ) : void
    {
        $ffi= $this->ffi;

        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);

        // Check Buffer A and B
        if($trans==BLASIF::NoTrans || $trans==BLASIF::ConjNoTrans) {
            $rows = $n; $cols = $k;
        } else if($trans==BLASIF::Trans || $trans==BLASIF::ConjTrans) {
            $rows = $k; $cols = $n;
        } else {
            throw new InvalidArgumentException('unknown transpose mode for bufferA.');
        }
        $this->assert_matrix_buffer_spec("A", $A, $rows, $cols, $offsetA, $ldA);
        $this->assert_matrix_buffer_spec("B", $B, $rows, $cols, $offsetB, $ldB);

        // Check Buffer C
        $this->assert_matrix_buffer_spec("C", $C, $n, $n, $offsetC, $ldC);

        // Check Buffer A and B and C
        $dtype = $A->dtype();
        if($dtype!=$B->dtype() || $dtype!=$C->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B and C");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_ssyr2k(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dsyr2k(
                    $order,
                    $uplo,
                    $trans,
                    $n,$k,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB,
                    $beta,
                    $C->addr($offsetC),$ldC);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function trmm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $B, int $offsetB, int $ldB) : void
    {
        $ffi= $this->ffi;
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        // Check Buffer A
        if($side==BLASIF::Left) {
            $sizeA = $m;
        } else if($side==BLASIF::Right) {
            $sizeA = $n;
        } else {
            throw new InvalidArgumentException('unknown transpose mode for bufferA.');
        }
        $this->assert_matrix_buffer_spec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);

        // Check Buffer B
        $this->assert_matrix_buffer_spec("B", $B, $m, $n, $offsetB, $ldB);

        // Check Buffer A and B
        $dtype = $A->dtype();
        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_strmm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dtrmm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }

    public function trsm(
        int $order,
        int $side,
        int $uplo,
        int $trans,
        int $diag,
        int $m,
        int $n,
        float $alpha,
        BufferInterface $A, int $offsetA, int $ldA,
        BufferInterface $B, int $offsetB, int $ldB) : void
    {
        $ffi= $this->ffi;
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        // Check Buffer A
        if($side==BLASIF::Left) {
            $sizeA = $m;
        } else if($side==BLASIF::Right) {
            $sizeA = $n;
        } else {
            throw new InvalidArgumentException('unknown transpose mode for bufferA.');
        }
        $this->assert_matrix_buffer_spec("A", $A, $sizeA, $sizeA, $offsetA, $ldA);

        // Check Buffer B
        $this->assert_matrix_buffer_spec("B", $B, $m, $n, $offsetB, $ldB);

        // Check Buffer A and B
        $dtype = $A->dtype();
        if($dtype!=$B->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and B");
        }

        switch($dtype) {
            case NDArray::float32:{
                $ffi->cblas_strsm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB);
                break;
            }
            case NDArray::float64:{
                $ffi->cblas_dtrsm(
                    $order,
                    $side,
                    $uplo,
                    $trans,
                    $diag,
                    $m,$n,
                    $alpha,
                    $A->addr($offsetA),$ldA,
                    $B->addr($offsetB),$ldB);
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
    }
}
