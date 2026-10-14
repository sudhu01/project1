// HTTP proxy for the registered orders-to-payment path. Admin state stays private.
package main

import (
    "context"
    "encoding/json"
    "flag"
    "net/http"
    "net/http/httputil"
    "net/url"
    "os"
    "os/signal"
    "strconv"
    "sync/atomic"
    "syscall"
    "time"
)

type delayingTransport struct { delay *atomic.Int64; requests *atomic.Int64 }
func (transport delayingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
    transport.requests.Add(1)
    milliseconds := transport.delay.Load()
    if milliseconds > 0 {
        select { case <-request.Context().Done(): return nil, request.Context().Err(); case <-time.After(time.Duration(milliseconds)*time.Millisecond): }
    }
    return http.DefaultTransport.RoundTrip(request)
}

func main() {
    mode := flag.String("mode", "serve", "serve, set or inspect")
    delayValue := flag.Int64("delay-ms", 0, "bounded directional delay")
    flag.Parse()
    if *delayValue < 0 || *delayValue > 1000 { os.Exit(2) }
    if *mode != "serve" {
        method := "GET"
        if *mode == "set" { method = "POST" } else if *mode != "inspect" { os.Exit(2) }
        req, _ := http.NewRequest(method, "http://127.0.0.1:8474/?delay="+strconv.FormatInt(*delayValue,10), nil)
        client := http.Client{Timeout: 3*time.Second}
        response, err := client.Do(req)
        if err != nil || response.StatusCode != 200 { os.Exit(3) }
        defer response.Body.Close()
        var payload map[string]int64
        json.NewDecoder(response.Body).Decode(&payload)
        json.NewEncoder(os.Stdout).Encode(payload)
        return
    }
    var delay, requests atomic.Int64
    upstream, _ := url.Parse("http://payment:80")
    proxy := httputil.NewSingleHostReverseProxy(upstream)
    proxy.Transport = delayingTransport{&delay, &requests}
    proxyServer := &http.Server{Addr: ":80", Handler: proxy, ReadHeaderTimeout: 3*time.Second}
    admin := &http.Server{Addr: "127.0.0.1:8474", ReadHeaderTimeout: 3*time.Second,
        Handler: http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
            if request.Method == "POST" {
                value, err := strconv.ParseInt(request.URL.Query().Get("delay"),10,64)
                if err != nil || value < 0 || value > 1000 { http.Error(writer,"invalid delay",400); return }
                delay.Store(int64(value))
                // Impairments expire even if the controller is interrupted.
                if value > 0 { time.AfterFunc(240*time.Second, func(){ delay.Store(0) }) }
            }
            json.NewEncoder(writer).Encode(map[string]int64{"delay_ms":delay.Load(), "forwarded_requests":requests.Load()})
        })}
    signals := make(chan os.Signal,1)
    signal.Notify(signals,syscall.SIGTERM,syscall.SIGINT)
    go admin.ListenAndServe()
    go func(){ <-signals; context, cancel := context.WithTimeout(context.Background(),3*time.Second); defer cancel(); proxyServer.Shutdown(context); admin.Shutdown(context) }()
    proxyServer.ListenAndServe()
}
