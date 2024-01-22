<?php
namespace Rindow\OpenBLAS\FFI;

use FFI;
use FFI\Env\Runtime as FFIEnvRuntime;
use FFI\Env\Status as FFIEnvStatus;
use FFI\Location\Locator as FFIEnvLocator;

class OpenBLASFactory
{
    private static ?FFI $ffi = null;
    protected array $libs = ['libopenblas.dll','libopenblas.so'];

    public function __construct(
        string $headerFile=null,
        array $libFiles=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        $headerFile = $headerFile ?? __DIR__ . "/openblas_win.h";
        $libFiles = $libFiles ?? $this->libs;
        $code = file_get_contents($headerFile);
        $pathname = FFIEnvLocator::resolve(...$libFiles);
        if($pathname) {
            $ffi = FFI::cdef($code,$pathname);
            self::$ffi = $ffi;
        }
    }

    public function isAvailable() : bool
    {
        $isAvailable = FFIEnvRuntime::isAvailable();
        if(!$isAvailable) {
            return false;
        }
        $pathname = FFIEnvLocator::resolve(...$this->libs);
        return $pathname!==null;
    }

    public function Blas() : Blas
    {
        return new Blas(self::$ffi);
    }

    public function Lapack() : Lapack
    {
        return new Lapack(self::$ffi);
    }
}
