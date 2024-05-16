<?php
namespace Rindow\OpenBLAS\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use InvalidArgumentException;
use RuntimeException;
use FFI;

use Interop\Polite\Math\Matrix\LinearBuffer as BufferInterface;

class ffi_char_t2
{
    public string $cdata;
}

class Lapackb
{
    use Utils;

    const LAPACK_WORK_MEMORY_ERROR      = -1010;
    const LAPACK_TRANSPOSE_MEMORY_ERROR = -1010;
    const LAPACK_ROW_MAJOR = 101;
    const LAPACK_COL_MAJOR = 102;

    protected FFI $ffi;

    public function __construct(FFI $ffi)
    {
        $this->ffi = $ffi;
    }

    public function gesvd(
        int $matrix_layout,
        int $jobu,
        int $jobvt,
        int $m,
        int $n,
        BufferInterface $A,  int $offsetA,  int $ldA,
        BufferInterface $S,  int $offsetS,
        BufferInterface $U,  int $offsetU,  int $ldU,
        BufferInterface $VT, int $offsetVT, int $ldVT,
        BufferInterface $SuperB,  int $offsetSuperB
    ) : void
    {
        $ffi = $this->ffi;
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        if( $offsetS < 0 ) {
            throw new InvalidArgumentException("offsetS must be greater than zero or equal");
        }
        if( $offsetU < 0 ) {
            throw new InvalidArgumentException("offsetU must be greater than zero or equal", 0);
        }
        if( $ldU <= 0 ) {
            throw new InvalidArgumentException("ldU must be greater than zero", 0);
        }
        if( $offsetVT < 0 ) {
            throw new InvalidArgumentException("offsetVT must be greater than zero or equal", 0);
        }
        if( $ldVT <= 0 ) {
            throw new InvalidArgumentException("ldVT must be greater than zero", 0);
        }
        if( $offsetSuperB < 0 ) {
            throw new InvalidArgumentException("offsetVT must be greater than zero or equal", 0);
        }
        // Check Buffer A
        $this->assert_matrix_buffer_spec("BufferA", $A,$m,$n,$offsetA,$ldA);
    
        // Check Buffer S
        if( $offsetS+min($m,$n) > count($S)) {
            throw new InvalidArgumentException("BufferS size is too small", 0);
        }
    
        // Check Buffer U
        if( $offsetU+$m*$ldU > count($U)) {
            throw new InvalidArgumentException("BufferU size is too small", 0);
        }
    
        // Check Buffer VT
        if( $offsetVT+$ldVT*$n > count($VT)) {
            throw new InvalidArgumentException("BufferVT size is too small", 0);
        }
    
        // Check Buffer SuperB
        if( $offsetSuperB+min($m,$n)-1 > count($SuperB)) {
            throw new InvalidArgumentException("bufferSuperB size is too small", 0);
        }
    
        $dtype = $A->dtype();
        // Check Buffer A and B and C
        if($dtype!=$S->dtype() ||
            $dtype!=$U->dtype() ||
            $dtype!=$VT->dtype() ||
            $dtype!=$SuperB->dtype()
        ) {
            throw new InvalidArgumentException("Unmatch data type", 0);
        }
        if($matrix_layout==self::LAPACK_ROW_MAJOR) {
            $targetA = clone $A;
            for($i=0;$i<$m;$i++) {
                for($j=0;$j<$n;$j++) {
                    $targetA[$j*$m+$i] = $A[$i*$n+$j];
                }
            }
            $targetU = clone $U;
            $targetVT = clone $VT;
            $ldA = $m;
        } elseif($matrix_layout==self::LAPACK_COL_MAJOR) {
            $targetA = $A;
            $targetU = $U;
            $targetVT = $VT;
        } else {
            throw new InvalidArgumentException("Invalid matrix_layout: $matrix_layout");
        }


        /** @var ffi_char_t2 $jobu_p */
        $jobu_p = $ffi->new('char[4]');
        $jobu_p[0] = chr($jobu);
        /** @var ffi_char_t2 $jobvt_p */
        $jobvt_p = $ffi->new('char[4]');
        $jobvt_p[0] = chr($jobvt);
        $m_p = $ffi->new('lapack_int[1]');
        $m_p[0] = $m;
        $n_p = $ffi->new('lapack_int[1]');
        $n_p[0] = $n;
        $ldA_p = $ffi->new('lapack_int[1]');
        $ldA_p[0] = $ldA;
        $ldU_p = $ffi->new('lapack_int[1]');
        $ldU_p[0] = $ldU;
        $ldVT_p = $ffi->new('lapack_int[1]');
        $ldVT_p[0] = $ldVT;
        $lwork_p = $ffi->new("lapack_int[1]");
        $lwork_p[0] = -1;
        $info_p = $ffi->new("lapack_int[1]");
        $info_p[0] = 0;

        //var_dump($ffi::sizeof($m_p)*8);
        //return;
        switch ($dtype) {
            case NDArray::float32:
                $wkopt_p = $ffi->new("float[1]");
                $ffi->sgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA->addr($offsetA), $ldA_p,
                    $S->addr($offsetS),
                    $targetU->addr($offsetU), $ldU_p,
                    $targetVT->addr($offsetVT), $ldVT_p,
                    $wkopt_p,$lwork_p,
                    $info_p
                );
                break;
            case NDArray::float64:
                $wkopt_p = $ffi->new("double[1]");
                $ffi->dgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA->addr($offsetA), $ldA_p,
                    $S->addr($offsetS),
                    $targetU->addr($offsetU), $ldU_p,
                    $targetVT->addr($offsetVT), $ldVT_p,
                    $wkopt_p,$lwork_p,
                    $info_p
                );
                break;
            default:
                throw new RuntimeException("Unsupported data type.", 0);
        }
        $info = $info_p[0];
        if( $info == self::LAPACK_WORK_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to allocate work array.", $info);
        } else if( $info == self::LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to transpose matrix.", $info);
        } else if( $info < 0 ) {
            throw new RuntimeException( "Wrong parameter. error=$info", $info);
        }
        $info_p[0] = 0;
        $lwork = (int)$wkopt_p[0];
        $lwork_p[0]= $lwork;

        switch ($dtype) {
            case NDArray::float32:
                $work = $ffi->new("float[$lwork]");
                $ffi->sgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA->addr($offsetA), $ldA_p,
                    $S->addr($offsetS),
                    $targetU->addr($offsetU), $ldU_p,
                    $targetVT->addr($offsetVT), $ldVT_p,
                    $work,$lwork_p,
                    $info_p
                );
                break;
            case NDArray::float64:
                $work = $ffi->new("double[$lwork]");
                $ffi->dgesvd_(
                    $jobu_p,
                    $jobvt_p,
                    $m_p,$n_p,
                    $targetA->addr($offsetA), $ldA_p,
                    $S->addr($offsetS),
                    $targetU->addr($offsetU), $ldU_p,
                    $targetVT->addr($offsetVT), $ldVT_p,
                    $work,$lwork_p,
                    $info_p
                );
                break;
            default:
                throw new RuntimeException("Unsupported data type.", 0);
        }
        $info = $info_p[0];

        if( $info == self::LAPACK_WORK_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to allocate work array.", $info);
        } else if( $info == self::LAPACK_TRANSPOSE_MEMORY_ERROR ) {
            throw new RuntimeException( "Not enough memory to transpose matrix.", $info);
        } else if( $info < 0 ) {
            throw new RuntimeException( "Wrong parameter. error=$info", $info);
        }

        $len = count($SuperB);
        for($i=0; $i<$len; $i++ ) {
            $SuperB[$i] = $work[$i+1];
        }
        if($matrix_layout==self::LAPACK_ROW_MAJOR) {
            for($i=0;$i<$m;$i++) {
                for($j=0;$j<$m;$j++) {
                    $U[$j*$m+$i] = $targetU[$i*$m+$j];
                }
            }
            for($i=0;$i<$n;$i++) {
                for($j=0;$j<$n;$j++) {
                    $VT[$j*$n+$i] = $targetVT[$i*$n+$j];
                }
            }
        }
    }
}
