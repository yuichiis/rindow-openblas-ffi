<?php
namespace Rindow\OpenBLAS\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS as BLASIF;
use InvalidArgumentException;

use Interop\Polite\Math\Matrix\LinearBuffer as BufferInterface;

trait Utils
{
    protected function assert_shape_parameter(
        string $name, int $n) : void
    {
        if($n<1) {
            throw new InvalidArgumentException("Argument $name must be greater than 0.");
        }
    }

    protected function assert_vector_buffer_spec(
        string $name, BufferInterface $buffer, int $n, int $offset, int $inc) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($inc<1) {
            throw new InvalidArgumentException("Argument inc$name must be greater than 0.");
        }
        if($offset+($n-1)*$inc >= count($buffer)) {
            throw new InvalidArgumentException("Vector specification too large for buffer$name.");
        }
    }

    protected function assert_matrix_buffer_spec(
        string $name, BufferInterface $buffer,
        int $m, int $n, int $offset, int $ld) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($offset+($m-1)*$ld+($n-1) >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }

    protected function assert_buffer_size(
        BufferInterface $buffer,
        int $offset, int $size,
        string $message) : void
    {
        if($size<1 || $offset<0 || count($buffer) < $offset+$size) {
            throw new InvalidArgumentException($message);
        }
    }

    private function assert_transpose_parameter(int $trans, string $name) : void
    {
        if($trans!==BLASIF::NoTrans && $trans!==BLASIF::ConjNoTrans &&
            $trans!==BLASIF::Trans && $trans!==BLASIF::ConjTrans) {
            throw new InvalidArgumentException("unknown transpose mode for buffer$name: $trans");
        }
        if($trans==BLASIF::ConjNoTrans && $this->isVecib()) {
            throw new InvalidArgumentException("Unsupported dtype on MacOS: {$trans}");
        }
    }

    private function assert_band_parameter(string $name, int $k) : void
    {
        if($k<0) {
            throw new InvalidArgumentException("Argument $name must be greater than equals 0.");
        }
    }

    private function assert_band_matrix_buffer_spec(
        string $name, BufferInterface $buffer,
        int $rows, int $cols, int $offset, int $ld, int $order) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($ld<$rows) {
            throw new InvalidArgumentException("Argument ld$name must be greater than or equals to $rows.");
        }
        if($order==BLASIF::RowMajor) {
            $last = ($rows-1)*$ld+($cols-1);
        } else {
            $last = ($cols-1)*$ld+($rows-1);
        }
        if($offset+$last >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }

    private function assert_matrix_buffer_spec_with_order(
        string $name, BufferInterface $buffer,
        int $m, int $n, int $offset, int $ld, int $order) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        if($ld<1) {
            throw new InvalidArgumentException("Argument ld$name must be greater than 0.");
        }
        if($order==BLASIF::RowMajor) {
            $last = ($m-1)*$ld+($n-1);
        } else {
            $last = ($n-1)*$ld+($m-1);
        }
        if($offset+$last >= count($buffer)) {
            throw new InvalidArgumentException("Matrix specification too large for buffer$name.");
        }
    }

    private function assert_packed_matrix_buffer_spec(
        string $name, BufferInterface $buffer, int $n, int $offset) : void
    {
        if($offset<0) {
            throw new InvalidArgumentException("Argument offset$name must be greater than equals 0.");
        }
        $size = intdiv($n*($n+1),2);
        if($offset+$size > count($buffer)) {
            throw new InvalidArgumentException("Packed matrix specification too large for buffer$name.");
        }
    }

    protected function toComplex(float|object $from,int $dtype, string $name) : object
    {
        if(is_numeric($from)) {
            throw new InvalidArgumentException("$name is not complex");
        }
        $ffi = $this->ffi;
        switch($dtype) {
            case NDArray::complex64: {
                $to = $ffi->new('openblas_complex_float');
                $to->real = $from->real;
                $to->imag = $from->imag;
                break;
            }
            case NDArray::complex128: {
                $to = $ffi->new('openblas_complex_double');
                $to->real = $from->real;
                $to->imag = $from->imag;
                break;
            }
            default: {
                throw new InvalidArgumentException('Unsuppored data type');
            }
        }
        return $to;
    }

}
