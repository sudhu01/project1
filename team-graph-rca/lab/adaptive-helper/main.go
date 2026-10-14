// Bounded CPU contention and retained allocation inside an approved lab container.
package main

import (
    "flag"
    "fmt"
    "os"
    "os/signal"
    "runtime"
    "regexp"
    "strconv"
    "sync"
    "syscall"
    "time"
)

func main() {
    mode := flag.String("mode", "cpu", "cpu or memory")
    seconds := flag.Int("seconds", 120, "bounded lifetime")
    workers := flag.Int("workers", 4, "CPU workers")
    bytes := flag.Int64("bytes", 0, "retained allocation")
    pidfile := flag.String("pidfile", "/tmp/rca-helper.pid", "private controller receipt")
    lease := flag.String("lease", "", "controller lease UUID")
    flag.Parse()
    if *seconds < 1 || *seconds > 240 || *workers < 1 || *workers > 16 || *bytes < 0 || *bytes > 512*1024*1024 || *pidfile != "/tmp/rca-helper.pid" || !regexp.MustCompile("^[a-f0-9-]{36}$").MatchString(*lease) || (*mode != "cpu" && *mode != "memory") || (*mode == "memory" && *bytes == 0) {
        fmt.Fprintln(os.Stderr, "invalid bounded helper settings")
        os.Exit(2)
    }
    file, err := os.OpenFile(*pidfile, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
    if err != nil { os.Exit(3) }
    fmt.Fprintln(file, strconv.Itoa(os.Getpid()))
    fmt.Fprintln(file, *lease)
    file.Close()
    defer os.Remove(*pidfile)
    stop := make(chan struct{})
    signals := make(chan os.Signal, 1)
    signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
    go func() { select { case <-signals: case <-time.After(time.Duration(*seconds)*time.Second): }; close(stop) }()
    switch *mode {
    case "cpu":
        runtime.GOMAXPROCS(*workers)
        var group sync.WaitGroup
        for i := 0; i < *workers; i++ {
            group.Add(1)
            go func() {
                defer group.Done()
                value := uint64(1)
                for { select { case <-stop: runtime.KeepAlive(value); return; default: }
                    for j:=0; j<100000; j++ { value = value*1664525+1013904223 }
                }
            }()
        }
        group.Wait()
    case "memory":
        retained := make([][]byte, 0)
        for allocated := int64(0); allocated < *bytes; {
            chunkSize := int64(1024*1024)
            if chunkSize > *bytes-allocated { chunkSize = *bytes-allocated }
            chunk := make([]byte, int(chunkSize))
            for i := 0; i < len(chunk); i += 4096 { chunk[i] = 1 }
            retained = append(retained, chunk)
            allocated += chunkSize
            select { case <-stop: runtime.KeepAlive(retained); return; case <-time.After(20*time.Millisecond): }
        }
        <-stop
        runtime.KeepAlive(retained)
    default:
        fmt.Fprintln(os.Stderr, "unsupported helper mode")
        os.Exit(2)
    }
}
