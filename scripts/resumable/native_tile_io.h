// Bounded known-demand prefetch. One worker, two owned aligned buffers.
// No route prediction; a consumer must release a buffer before it is reused.
#pragma once
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

class NativeTileIO {
public:
    struct Read { size_t offset, payload, count; };
private:
    int fd;
    const std::vector<Read> jobs;
    std::array<void *,2> buffers{};
    std::mutex mutex;
    std::condition_variable changed;
    size_t consumed=0,completed=0;
    bool stopping=false;
    std::exception_ptr error;
    std::thread worker;
    uint64_t bytes=0,calls=0;
    double read_seconds=0,wait_seconds=0;
    void run() noexcept {
        try {
            for(size_t i=0;i<jobs.size();++i) {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock,[&]{return stopping||i<consumed+buffers.size();});
                    if(stopping)return;
                }
                const auto & job=jobs[i];
                auto start=std::chrono::steady_clock::now();ssize_t got;
                do { got=pread(fd,buffers[i%2],job.count,job.offset); } while(got<0&&errno==EINTR);
                read_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                if(got<0||size_t(got)<job.payload)throw std::runtime_error("short asynchronous tile read");
                bytes+=got;++calls;
                {
                    std::lock_guard<std::mutex> lock(mutex);completed=i+1;
                }
                changed.notify_all();
            }
        } catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);error=std::current_exception();}
            changed.notify_all();
        }
    }
public:
    NativeTileIO(int source,std::vector<Read> reads,size_t capacity):fd(source),jobs(std::move(reads)) {
        for(const auto & j:jobs)if(!j.payload||j.payload>j.count||j.count>capacity||j.offset%4096||j.count%4096)
            throw std::runtime_error("invalid asynchronous tile extent");
        try {
            for(auto & b:buffers)if(posix_memalign(&b,4096,capacity))throw std::runtime_error("tile pipeline allocation failed");
            worker=std::thread([this]{run();});
        } catch(...) {for(auto b:buffers)free(b);throw;}
    }
    NativeTileIO(const NativeTileIO&)=delete;
    ~NativeTileIO() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        changed.notify_all();if(worker.joinable())worker.join();
        for(auto b:buffers)free(b);
    }
    const void * acquire(size_t index) {
        auto start=std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex);
        if(index!=consumed||index>=jobs.size())throw std::runtime_error("invalid tile consumption order");
        changed.wait(lock,[&]{return completed>index||error;});
        wait_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(error)std::rethrow_exception(error);
        return buffers[index%2];
    }
    void release(size_t index) {
        {std::lock_guard<std::mutex> lock(mutex);
         if(index!=consumed||completed<=index)throw std::runtime_error("invalid tile release");
         ++consumed;}
        changed.notify_all();
    }
    void finish(uint64_t & total_bytes,uint64_t & total_calls,double & total_read,double & total_wait) {
        if(consumed!=jobs.size())throw std::runtime_error("unfinished tile pipeline");
        if(worker.joinable())worker.join();
        if(error)std::rethrow_exception(error);
        total_bytes+=bytes;total_calls+=calls;total_read+=read_seconds;total_wait+=wait_seconds;
    }
};
