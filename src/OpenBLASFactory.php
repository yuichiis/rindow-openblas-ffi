<?php
namespace Rindow\OpenBLAS\FFI;

use FFI;
//use FFI\Env\Runtime as FFIEnvRuntime;
//use FFI\Env\Status as FFIEnvStatus;
//use FFI\Location\Locator as FFIEnvLocator;
use FFI\Exception as FFIException;
use RuntimeException;

class OpenBLASFactory
{
    private static ?FFI $ffi = null;
    private static ?FFI $ffiLapacke = null;
    private static ?FFI $ffiLapack = null;
    protected array $configMatrix = [
        'WINNT' => [
            'blas' => [
                'header' => __DIR__.'/openblas.h',
                'libs' => ['libopenblas.dll'],
            ],
            'lapacke' => [
                'header' => __DIR__ . '/lapacke.h',
                'libs' => ['libopenblas.dll'],
            ],
            'lapack' => [
                'header' => __DIR__ . '/lapack.h',
                'libs' => ['libopenblas.dll'],
            ],
        ],
        'Linux' => [
            'blas' => [
                'header' => __DIR__.'/openblas.h',
                'libs' => ['libopenblas.so.0'],
            ],
            'lapacke' => [
                'header' => __DIR__ . '/lapacke.h',
                'libs' => ['liblapacke.so.3'],
            ],
            'lapack' => [
                'header' => __DIR__ . '/lapack.h',
                'libs' => ['libopenblas.so.0'],
            ],
        ],
        'Darwin' => [
            'blas' => [
                'header' => __DIR__.'/cblas_new_vecLib.h',
                'libs' => ['/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib'],
            ],
            'lapacke' => [
                'header' => null,
                'libs' => null,
            ],
            'lapack' => [
                'header' => __DIR__ . '/clapack_vecLib.h',
                'libs' => ['/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/vecLib'],
            ],
        ],
    ];
    /** @var array<string> $errors */
    private array $errors = [];

    /**
     * @param array<string> $libFiles
     * @param array<string> $lapackeLibs
     */
    public function __construct(
        string $headerFile=null,
        array $libFiles=null,
        string $lapackeHeader=null,
        array $lapackeLibs=null,
        )
    {
        if(self::$ffi!==null) {
            return;
        }
        if(!extension_loaded('ffi')) {
            return;
        }

        $config = $this->generateConfig([
            'blas' => [
                'header' => $headerFile,
                'libs' => $libFiles,
            ],
            'lapacke' => [
                'header' => $lapackeHeader,
                'libs' => $lapackeLibs,
            ],
            'lapack' => [],
        ]);
        $drivers = $this->loadLibraries($config);
        if(isset($drivers['blas'])) {
            self::$ffi = $drivers['blas'];
        }
        if(isset($drivers['lapacke'])) {
            self::$ffiLapacke = $drivers['lapacke'];
        }
        if(isset($drivers['lapack'])) {
            self::$ffiLapack = $drivers['lapack'];
        }
        return;
    
        //
        // blas
        //
        if(PHP_OS=='Darwin') {
            $headerFile = $headerFile ?? __DIR__.'/cblas_new_vecLib.h';
        } else {
            $headerFile = $headerFile ?? __DIR__.'/openblas.h';
        }
        if($libFiles==null) {
            if(PHP_OS=='Linux') {
                $libFiles = $this->libs_linux;
            } elseif(PHP_OS=='Darwin') {
                $libFiles = $this->libs_mac;
            } elseif(PHP_OS=='WINNT') {
                $libFiles = $this->libs_win;
            } else {
                throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
            }
        }
        $code = file_get_contents($headerFile);
        // ***************************************************************
        // FFI Locator is incompletely implemented. It is often not found.
        // ***************************************************************
        //$pathname = FFIEnvLocator::resolve(...$libFiles);
        //if($pathname) {
        //    $ffi = FFI::cdef($code,$pathname);
        //    self::$ffi = $ffi;
        //}
        foreach ($libFiles as $filename) {
            $ffi = null;
            try {
                $ffi = FFI::cdef($code,$filename);
            } catch(FFIException $e) {
                echo "==== BLAS filename: ".$filename." ====\n";
                echo $e->getMessage()."\n";
                continue;
            }
            self::$ffi = $ffi;
            break;
        }
        //
        // lapacke
        //
        $lapackeHeader = $lapackeHeader ?? __DIR__ . '/lapacke.h';
        if($lapackeLibs==null) {
            if(PHP_OS=='Linux') {
                $lapackeLibs = $this->lapacke_linux;
            } elseif(PHP_OS=='Darwin') {
                $lapackeLibs = $this->lapacke_mac;
            } elseif(PHP_OS=='WINNT') {
                $lapackeLibs = $this->lapacke_win;
            } else {
                throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
            }
        }
        $code = file_get_contents($lapackeHeader);
        // ***************************************************************
        // FFI Locator is incompletely implemented. It is often not found.
        // ***************************************************************
        //if(PHP_OS=='Linux') {
        //    $libFiles = $this->lapackeLibs;
        //    $pathname = FFIEnvLocator::resolve(...$libFiles);
        //}
        //if($pathname) {
        //    $code = file_get_contents($headerFile);
        //    $ffi = FFI::cdef($code,$pathname);
        //    self::$ffiLapacke = $ffi;
        //}
        foreach ($lapackeLibs as $filename) {
            $ffiLapacke = null;
            try {
                $ffiLapacke = FFI::cdef($code,$filename);
            } catch(FFIException $e) {
                echo "==== LAPACKE filename: ".$filename." ====\n";
                echo $e->getMessage()."\n";
                continue;
            }
            self::$ffiLapacke = $ffiLapacke;
            break;
        }

        //
        // lapack
        //
        $lapackHeader = $lapackHeader ?? __DIR__ . '/lapack.h';
        if($lapackeLibs==null) {
            if(PHP_OS=='Linux') {
                $lapackeLibs = $this->lapacke_linux;
            } elseif(PHP_OS=='Darwin') {
                $lapackeLibs = $this->lapacke_mac;
            } elseif(PHP_OS=='WINNT') {
                $lapackeLibs = $this->lapacke_win;
            } else {
                throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
            }
        }
        $code = file_get_contents($lapackHeader);
        // ***************************************************************
        // FFI Locator is incompletely implemented. It is often not found.
        // ***************************************************************
        //if(PHP_OS=='Linux') {
        //    $libFiles = $this->lapackeLibs;
        //    $pathname = FFIEnvLocator::resolve(...$libFiles);
        //}
        //if($pathname) {
        //    $code = file_get_contents($headerFile);
        //    $ffi = FFI::cdef($code,$pathname);
        //    self::$ffiLapacke = $ffi;
        //}
        foreach ($lapackeLibs as $filename) {
            $ffiLapack = null;
            try {
                $ffiLapack = FFI::cdef($code,$filename);
            } catch(FFIException $e) {
                echo $e->getMessage()."\n";
                continue;
            }
            self::$ffiLapack = $ffiLapack;
            break;
        }


    }

    protected function generateConfig(array $params) : array
    {
        $os = PHP_OS;
        if(!isset($this->configMatrix[$os])) {
            throw new RuntimeException('Unknown operating system: "'.PHP_OS.'"');
        }
        $defaults = $this->configMatrix[$os];
        foreach($defaults as $type => $names) {
            foreach($names as $key => $value) {
                $params[$type][$key] ??= $value;
            }
        }
        return $params;
    }

    protected function loadLibraries(array $params) : array
    {
        /** @var array<objects> $ffis */
        $ffis = [];
        foreach($params as $key => $param) {
            $code = file_get_contents($param['header']);
            if($code===false) {
                throw new RuntimeException('The header file not found: "'.$name['header'].'"');
            }
            foreach($param['libs'] as $filename) {
                $ffi = null;
                try {
                    $ffi = FFI::cdef($code,$filename);
                } catch(FFIException $e) {
                    $this->errors[] = $e->getMessage();
                    continue;
                }
                $ffis[$key] = $ffi;
                break;
            }
        }
        return $ffis;
    }

    public function isAvailable() : bool
    {
        return self::$ffi!==null;
        //$isAvailable = FFIEnvRuntime::isAvailable();
        //if(!$isAvailable) {
        //    return false;
        //}
        //$pathname = FFIEnvLocator::resolve(...$this->libs);
        //return $pathname!==null;
    }

    public function Blas() : Blas
    {
        if(self::$ffi==null) {
            throw new RuntimeException('openblas library not loaded.');
        }
        return new Blas(self::$ffi);
    }

    public function Lapack() : Lapack
    {
        if(self::$ffiLapacke==null) {
            throw new RuntimeException('lapacke library not loaded.');
        }
        return new Lapack(self::$ffiLapacke);
    }

    public function Lapackb() : Lapackb
    {
        if(self::$ffiLapack==null) {
            throw new RuntimeException('lapacke library not loaded.');
        }
        return new Lapackb(self::$ffiLapack);
    }
}
