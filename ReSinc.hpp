#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define _USE_MATH_DEFINES

// =============================================================================
//  TEMPLATE UTILITIES
// =============================================================================

/** * @brief Helper to create a non-deducible context.
 * This forces the compiler to deduce TYPE from other arguments (like buffers)
 * and then simply convert the sample rate arguments to that TYPE.
 */
template<typename T>
struct identity {
    using type = T;
};
template<typename T>
using non_deduced = typename identity<T>::type;

// =============================================================================
//  SFINAE TRAITS
// =============================================================================

/**
 * @brief Internal traits to detect buffer capabilities at compile-time.
 * * These structures are used by the Oversampler to automatically select the
 * correct overload for interpolate/decimate based on the input type.
 */
namespace resinc_traits {
    // Helper: Detects if T has .data() and .size() methods
    template<typename T, typename = void>
    struct has_size_and_data : std::false_type {};
    template<typename T>
    struct has_size_and_data<T,
                             std::void_t<decltype(std::declval<T>().data()),
                                         decltype(std::declval<T>().size())>> :
        std::true_type {};

    /** @brief Detects single-channel containers (e.g. std::vector<float>). */
    template<typename T, typename = void>
    struct is_single_channel : std::false_type {};
    template<typename T>
    struct is_single_channel<
        T,
        std::enable_if_t<has_size_and_data<T>::value &&
                         std::is_arithmetic_v<
                             std::decay_t<decltype(std::declval<T>()[0])>>>> :
        std::true_type {};

    /** @brief Detects multi-channel containers (e.g.
     * std::vector<std::vector<float>>). */
    template<typename T, typename = void>
    struct is_multi_channel : std::false_type {};
    template<typename T>
    struct is_multi_channel<
        T,
        std::enable_if_t<has_size_and_data<T>::value &&
                         has_size_and_data<std::decay_t<
                             decltype(std::declval<T>()[0])>>::value>> :
        std::true_type {};

    /** @brief Detects JUCE-compatible AudioBuffer classes (has
     * getReadPointer/getWritePointer). */
    template<typename T, typename = void>
    struct is_juce_type : std::false_type {};
    template<typename T>
    struct is_juce_type<
        T,
        std::void_t<
            // Method Existence
            decltype(std::declval<const T>().getNumChannels()),
            decltype(std::declval<const T>().getNumSamples()),
            decltype(std::declval<const T>().getReadPointer(0)),
            decltype(std::declval<T>().getWritePointer(0)),
            decltype(std::declval<T>().clear()),
            std::enable_if_t<std::is_integral_v<
                decltype(std::declval<T>().getNumChannels())>>,
            std::enable_if_t<std::is_pointer_v<
                decltype(std::declval<T>().getReadPointer(0))>>>> :
        std::true_type {};
}    // namespace resinc_traits

// =============================================================================
//  COMPONENTS
// =============================================================================

/**
 * @brief Fixed-size circular (ring) buffer.
 *
 * @tparam TYPE    Element type stored in the buffer.
 * @tparam size    Number of elements the buffer holds.
 *
 * Behaviour:
 * - push(element) inserts a new element and overwrites the oldest when full.
 * - operator[](index) accesses elements relative to the oldest element.
 * * Complexity: O(1) for all operations.
 */
template<class TYPE, size_t size>
class CircularBuffer {
private:
    std::array<TYPE, size> buf;
    size_t oldestIndex;

public:
    CircularBuffer(TYPE initValue);
    void push(TYPE element);
    TYPE& operator[](size_t index);
    void clear();
};

namespace Window {
    /**
     * @brief Generic window function base class.
     * Provides accessors and an applyOn() helper to multiply an array by the
     * window.
     */
    template<typename TYPE, size_t N>
    class Window {
    public:
        virtual ~Window() = default;
        virtual TYPE& operator[](size_t i) = 0;
        virtual const TYPE& operator[](size_t i) const = 0;
        void applyOn(std::array<TYPE, N>& data) const;
        size_t size() const;

    protected:
        Window() = default;
    };

    /**
     * @brief Kaiser window implementation.
     * parameterised by beta to control side-lobe attenuation.
     */
    template<typename TYPE, size_t N>
    class Kaiser : public Window<TYPE, N> {
    public:
        Kaiser(TYPE beta);
        TYPE& operator[](size_t i) override;
        const TYPE& operator[](size_t i) const override;

    private:
        TYPE besselI0(TYPE x);
        const TYPE beta;
        std::array<TYPE, (N + 1) / 2> window;
    };
}    // namespace Window

namespace internal {
    /**
     * @brief Internal Sinc lookup table.
     * Handles symmetric storage and fractional indexing.
     */
    template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
    class Sinc {
    private:
        std::array<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR> sinc;

    public:
        Sinc();
        TYPE& operator[](int i);
        TYPE& operator()(int i, int delta);
        void configure(TYPE sampleRate);
        void configure_with_cutoff(TYPE cutoff, TYPE sampleRate);
        size_t size() const;
        void applyWindow(
            const Window::Window<TYPE,
                                 (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2>&
                window);
    };

    /**
     * @brief Core implementation for stateless one-shot resampling.
     * * This helper performs a full sample rate conversion by locally
     * configuring a Sinc filter and Kaiser window. Unlike the class-based
     * resampler, this function treats the boundaries of the input buffer as
     * silence (zero-padding).
     * *
     * * Performance Note: This function re-calculates the filter table on every
     * call. For repeated block-based processing, use the Resampler class
     * instead.
     */
    template<typename TYPE, int SINC_RADIUS, int RESOLUTION = 256>
    int resample_helper(const TYPE* const* ptrToInBuffers,
                        TYPE* const* ptrToOutBuffers,
                        int numChannels,
                        int numSamples,
                        TYPE sourceSampleRate,
                        TYPE targetSampleRate);
}    // namespace internal

// =============================================================================
//  OVERSAMPLER CLASS
// =============================================================================

/**
 * @brief Multi-channel resampler (real-time / streaming-safe).
 *
 * Resample converts between sample rates by interpolating and decimating
 * buffers using a precomputed Sinc table. The class maintains per-channel
 * circular history buffers and is designed for continuous (real-time)
 * processing.
 *
 * Important notes:
 * - Internal buffering: The class maintains history across calls. Do not
 * assume independence; feed blocks consecutively.
 * - Latency: Effective latency is 2*N input samples (2 * SINC_RADIUS).
 *
 * Template parameters:
 * @tparam TYPE     Sample type (float, double).
 * @tparam OVERSAMPLE_FACTOR Interpolation factor (e.g., 2x, 4x).
 * @tparam N     Sinc kernel radius (taps per side).
 */
template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
class Oversampler {
public:
    Oversampler();
    ~Oversampler() = default;
    Oversampler(const Oversampler&) = delete;
    Oversampler& operator=(const Oversampler&) = delete;
    Oversampler(Oversampler&&) noexcept = default;
    Oversampler& operator=(Oversampler&&) noexcept = default;

    /**
     * @brief Configures internal buffers for a specific session format.
     * Must be called before processing.
     * @param sampleRate The base sample rate of the input signal.
     * @param maxChannels Maximum number of channels expected.
     * @param maxBlockSize Maximum number of samples per input block.
     */
    void configure(TYPE sampleRate, int maxChannels, int maxBlockSize);

    // =========================================================================
    // INTERPOLATE (Upsample Input -> Internal Buffer)
    // =========================================================================

    /**
     * @brief Interpolates a JUCE-style AudioBuffer.
     * Accepts any class with getNumChannels(), getNumSamples(), and
     * getArrayOfReadPointers().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T>>::value>
        interpolate(T&& buffer);

    /**
     * @brief Interpolates a single-channel container.
     * Accepts std::vector, std::array, or custom containers with .data() and
     * .size().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_single_channel<std::decay_t<T>>::value>
        interpolate(T&& buffer);

    /**
     * @brief Interpolates a multi-channel container (vector of vectors).
     * Accepts std::vector<std::vector<float>>.
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_multi_channel<std::decay_t<T>>::value>
        interpolate(T&& buffer);

    /**
     * @brief Interpolates from raw pointers (Base Implementation).
     * @param ptrToBuffers Array of pointers to channel data [channel][sample].
     * @param numChannels Number of channels to read.
     * @param numSamples Number of input samples to read per channel.
     */
    void interpolate(const TYPE* const* ptrToBuffers,
                     int numChannels,
                     int numSamples);

    // =========================================================================
    // DECIMATE (Downsample Internal Buffer -> Output)
    // =========================================================================

    /**
     * @brief Decimates into a JUCE-style AudioBuffer.
     * Writes result to the provided buffer using getArrayOfWritePointers().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T>>::value>
        decimate(T& buffer);

    /**
     * @brief Decimates into a single-channel container.
     * Writes result directly to .data().
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_single_channel<std::decay_t<T>>::value>
        decimate(T& buffer);

    /**
     * @brief Decimates into a multi-channel container (vector of vectors).
     */
    template<typename T>
    typename std::enable_if_t<
        resinc_traits::is_multi_channel<std::decay_t<T>>::value>
        decimate(T& buffer);

    /**
     * @brief Decimates into raw pointers (Base Implementation).
     * @param ptrToBuffers Destination array of pointers [channel][sample].
     * @param numChannels Number of channels to write.
     * @param numSamples Number of samples to produce (target size).
     */
    void decimate(TYPE* const* ptrToBuffers, int numChannels, int numSamples);

    // =========================================================================
    // PROCESS CALLBACKS
    // =========================================================================

    /**
     * @brief Processes upsampled data using a JUCE-style wrapper.
     * * Constructs a temporary wrapper (e.g., juce::AudioBuffer) around the
     * internal high-rate buffers and passes it to the callback.
     * * @tparam T The buffer wrapper type to construct (e.g.,
     * juce::AudioBuffer<float>).
     * @tparam Callable Deduced type of the callback.
     * @param processBlock A callable receiving the constructed buffer:
     * void(T&).
     */
    template<typename T, typename Callable>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T>>::value>
        process(Callable&& processBlock);
    /**
     * @brief Processes upsampled data via the internal multi-channel container.
     * * Provides direct, automatic access to the internal
     * `vector<vector<TYPE>>`. Use this for algorithms requiring cross-channel
     * data (e.g., M/S or Stereo-linking).
     * * @tparam Callable Deduced type of the callback.
     * @param processBlock A callable receiving the internal buffer:
     * void(std::vector<std::vector<TYPE>>&).
     */
    template<typename Callable>
    void process(Callable&& processBlock);

    /**
     * @brief Processes upsampled data channel-by-channel.
     * * Iterates through the internal buffers and executes the callback for
     * each individual channel. Ideal for independent mono-style DSP effects.
     * * @tparam Callable Deduced type of the callback.
     * @param processBlock A callable receiving a single channel:
     * void(std::vector<TYPE>&).
     */
    template<typename Callable>
    void processEach(Callable&& processBlock);

private:
    internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS> sinc;
    std::vector<std::vector<TYPE>> interpolatedBuf;
    std::vector<CircularBuffer<TYPE, SINC_RADIUS>> beginBuf, endBuf;
    std::vector<CircularBuffer<TYPE, SINC_RADIUS * OVERSAMPLE_FACTOR>>
        decBeginBuf, decEndBuf;
    std::vector<std::vector<TYPE>> x_interp, x_decim;

    void interpolate_helper(const TYPE* const* ptrToBuffers,
                            int numChannels,
                            int numSamples);
    void decimate_helper(TYPE* const* ptrToBuffers,
                         int numChannels,
                         int numSamples);
};

// =============================================================================
//  RESAMPLER CLASS
// =============================================================================

/**
 * @brief Multi-channel asynchronous resampler (real-time / streaming-safe).
 *
 * Resampler converts between arbitrary sample rates using a precomputed
 * Sinc table with high-resolution phase lookup. Unlike the Oversampler,
 * it handles non-integer ratios and maintains internal phase state for
 * seamless multi-block processing.
 *
 * Important notes:
 * - Internal buffering: The class maintains history and phase across calls.
 * - Gain scaling: Automatically applies compensation when downsampling.
 *
 * Template parameters:
 * @tparam TYPE       Sample type (float, double).
 * @tparam SINC_RADIUS Sinc kernel radius (taps per side).
 * @tparam RESOLUTION  Fractional phase lookup resolution.
 */
template<typename TYPE, int SINC_RADIUS, int RESOLUTION = 256>
class Resampler {
public:
    Resampler();
    ~Resampler() = default;
    Resampler(const Resampler&) = delete;
    Resampler& operator=(const Resampler&) = delete;
    Resampler(Resampler&&) noexcept = default;
    Resampler& operator=(Resampler&&) noexcept = default;

    /**
     * @brief Configures internal buffers for arbitrary rate conversion.
     * Must be called before processing.
     * @param sampleRate The base sample rate of the input signal.
     * @param targetSampleRate The target output sample rate.
     * @param maxChannels Maximum number of channels expected.
     * @param maxBlockSize Maximum number of samples per input block.
     */
    void configure(TYPE sampleRate,
                   TYPE targetSampleRate,
                   int maxChannels,
                   int maxBlockSize);

    /**
     * @brief Resamples a JUCE-style AudioBuffer.
     * @return The number of output samples produced.
     */
    template<typename T1, typename T2>
    typename std::enable_if_t<
        resinc_traits::is_juce_type<std::decay_t<T1>>::value &&
            resinc_traits::is_juce_type<std::decay_t<T2>>::value,
        int>
        resample(T1&& input, T2& output);

    /**
     * @brief Resamples a single-channel container.
     * @return The number of output samples produced.
     */
    template<typename T1, typename T2>
    typename std::enable_if_t<
        resinc_traits::is_single_channel<std::decay_t<T1>>::value &&
            resinc_traits::is_single_channel<std::decay_t<T2>>::value,
        int>
        resample(T1&& input, T2& output);

    /**
     * @brief Resamples a multi-channel container (vector of vectors).
     * @return The number of output samples produced.
     */
    template<typename T1, typename T2>
    typename std::enable_if_t<
        resinc_traits::is_multi_channel<std::decay_t<T1>>::value &&
            resinc_traits::is_multi_channel<std::decay_t<T2>>::value,
        int>
        resample(T1&& input, T2& output);

    /**
     * @brief Resamples from raw pointers (Base Implementation).
     * @param ptrToInBuffers Array of pointers to input channel data.
     * @param ptrToBuffers Array of pointers to output channel data.
     * @param numChannels Number of channels to process.
     * @param numSamples Number of input samples to read.
     * @return The number of output samples produced.
     */
    int resample(const TYPE* const* ptrToInBuffers,
                 TYPE* const* ptrToBuffers,
                 int numChannels,
                 int numSamples);

private:
    internal::Sinc<TYPE, RESOLUTION, SINC_RADIUS> sinc;
    std::vector<CircularBuffer<TYPE, SINC_RADIUS>> beginBuf, endBuf;
    std::vector<std::vector<TYPE>> x;
    TYPE current_phase;
    TYPE maxOutputSize;
    TYPE oversampleFactor;
    int resample_helper(const TYPE* const* ptrToInBuffers,
                        TYPE* const* ptrToOutBuffers,
                        int numChannels,
                        int numSamples);
};

// =============================================================================
//  STATELESS RESAMPLING FUNCTIONS
// =============================================================================

/**
 * @brief Resamples a JUCE-style AudioBuffer without internal history.
 * * This is a one-shot function that ignores previous or future audio blocks.
 * Ideal for offline processing or UI assets.
 * * @tparam TYPE             Sample type (float, double).
 * @tparam SINC_RADIUS      Sinc kernel radius.
 * @tparam RESOLUTION       Phase lookup table resolution.
 * @return The number of output samples produced.
 */
template<typename TYPE,
         int SINC_RADIUS,
         int RESOLUTION = 256,
         typename T1,
         typename T2>
typename std::enable_if_t<
    resinc_traits::is_juce_type<std::decay_t<T1>>::value &&
        resinc_traits::is_juce_type<std::decay_t<T2>>::value,
    int>
    resample(T1&& input,
             T2& output,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate);

/**
 * @brief Resamples a single-channel container (e.g., std::vector<float>)
 * without internal history.
 */
template<typename TYPE,
         int SINC_RADIUS,
         int RESOLUTION = 256,
         typename T1,
         typename T2>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T1>>::value &&
        resinc_traits::is_single_channel<std::decay_t<T2>>::value,
    int>
    resample(T1&& input,
             T2& output,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate);

/**
 * @brief Resamples a multi-channel container (e.g., vector of vectors) without
 * internal history.
 */
template<typename TYPE,
         int SINC_RADIUS,
         int RESOLUTION = 256,
         typename T1,
         typename T2>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T1>>::value &&
        resinc_traits::is_multi_channel<std::decay_t<T2>>::value,
    int>
    resample(T1&& input,
             T2& output,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate);

/**
 * @brief Resamples from raw pointers without internal history (Base
 * Implementation).
 * * @param ptrToInBuffers    Array of pointers to input channel data.
 * @param ptrToOutBuffers   Array of pointers to output channel data.
 * @param numChannels       Number of channels to process.
 * @param numSamples        Number of input samples.
 * @param sourceSampleRate  The sample rate of the input data.
 * @param targetSampleRate  The desired output sample rate.
 * @return The number of output samples produced.
 */
template<typename TYPE, int SINC_RADIUS, int RESOLUTION = 256>
int resample(const TYPE* const* ptrToInBuffers,
             TYPE* const* ptrToOutBuffers,
             int numChannels,
             int numSamples,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate);

// =============================================================================
//  IMPLEMENTATIONS: CircularBuffer
// =============================================================================

template<class TYPE, size_t size>
CircularBuffer<TYPE, size>::CircularBuffer(TYPE initValue) : oldestIndex {0} {
    buf.fill(initValue);
}

template<class TYPE, size_t size>
void CircularBuffer<TYPE, size>::push(TYPE element) {
    this->buf[this->oldestIndex] = std::move(element);
    this->oldestIndex = this->oldestIndex == size - 1 ? 0 : oldestIndex + 1;
}

template<class TYPE, size_t size>
TYPE& CircularBuffer<TYPE, size>::operator[](size_t index) {
    return this->buf[(this->oldestIndex + index) % size];
}

template<class TYPE, size_t size>
void CircularBuffer<TYPE, size>::clear() {
    buf.fill(TYPE {0});
    this->oldestIndex = 0;
}

// =============================================================================
//  IMPLEMENTATIONS: Window
// =============================================================================

template<typename TYPE, size_t N>
size_t Window::Window<TYPE, N>::size() const {
    return N;
}

template<typename TYPE, size_t N>
void Window::Window<TYPE, N>::applyOn(std::array<TYPE, N>& data) const {
    for(size_t i = 0; i < N; i++) data[i] *= (*this)[i];
}

template<typename TYPE, size_t N>
Window::Kaiser<TYPE, N>::Kaiser(TYPE beta) : Window<TYPE, N> {}, beta {beta} {
    if(beta < TYPE(0.0))
        throw std::invalid_argument("Kaiser beta value must be non-negative.");
    const TYPE M_denominator = static_cast<TYPE>(N - 1);
    const TYPE den = besselI0(beta);
    const size_t storageSize = (N + 1) / 2;
    const TYPE center = static_cast<TYPE>(N - 1) / TYPE(2.0);
    for(size_t i = 0; i < storageSize; i++) {
        TYPE n = static_cast<TYPE>(i) - center;
        TYPE arg =
            beta * std::sqrt(TYPE(1.0) - std::pow(TYPE(2.0) * n / M_denominator,
                                                  TYPE(2.0)));
        window[i] = besselI0(arg) / den;
    }
}

template<typename TYPE, size_t N>
TYPE& Window::Kaiser<TYPE, N>::operator[](size_t i) {
    const size_t pivot = N / 2;
    return window[(i < pivot) ? i : (N - 1 - i)];
}

template<typename TYPE, size_t N>
const TYPE& Window::Kaiser<TYPE, N>::operator[](size_t i) const {
    const size_t pivot = N / 2;
    return window[(i < pivot) ? i : (N - 1 - i)];
}

template<typename TYPE, size_t N>
TYPE Window::Kaiser<TYPE, N>::besselI0(TYPE x) {
    const TYPE epsilon = TYPE(1e-6);
    TYPE sum = TYPE(1.0), term = TYPE(1.0), k = TYPE(1.0),
         factorial = TYPE(1.0);
    while(term > epsilon * sum) {
        factorial *= k;
        term = std::pow(x / TYPE(2.0), TYPE(2) * k) / (factorial * factorial);
        sum += term;
        k += TYPE(1.0);
    }
    return sum;
}

// =============================================================================
//  IMPLEMENTATIONS: Sinc
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Sinc() : sinc {0} {}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
TYPE& internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::operator[](int i) {
    return sinc[i < 0 ? -i : i];
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
TYPE& internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::operator()(
    int i,
    int delta) {
    if(i < 0) return (*this)[(-i) * OVERSAMPLE_FACTOR - delta];
    return (*this)[i * OVERSAMPLE_FACTOR + delta];
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::configure(
    TYPE sampleRate) {
    TYPE fc = sampleRate / TYPE(2);
    TYPE T_ = TYPE(1) / sampleRate;
    for(int i = 0; i <= SINC_RADIUS; i++) {
        for(int delta = 0; delta < OVERSAMPLE_FACTOR; delta++) {
            TYPE index = static_cast<TYPE>(i) +
                         (TYPE(1) / static_cast<TYPE>(OVERSAMPLE_FACTOR)) *
                             static_cast<TYPE>(delta);
            if(index == TYPE(0.0)) (*this)(i, delta) = TYPE(1.0);
            else
                (*this)(i, delta) = std::sin(TYPE(2) * M_PI * fc * index * T_) /
                                    (TYPE(2) * M_PI * fc * index * T_);
        }
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::
    configure_with_cutoff(TYPE cutoff, TYPE sampleRate) {
    if(cutoff <= TYPE(0.0) || cutoff > sampleRate / TYPE(2.0))
        throw std::invalid_argument(
            "Cutoff frequency must be between 0 and Nyquist frequency.");
    TYPE T_ = TYPE(1) / sampleRate;
    for(int i = 0; i <= SINC_RADIUS; i++) {
        for(int delta = 0; delta < OVERSAMPLE_FACTOR; delta++) {
            TYPE index = static_cast<TYPE>(i) +
                         (TYPE(1) / static_cast<TYPE>(OVERSAMPLE_FACTOR)) *
                             static_cast<TYPE>(delta);
            if(index == TYPE(0.0)) (*this)(i, delta) = TYPE(1.0);
            else
                (*this)(i, delta) =
                    std::sin(TYPE(2) * M_PI * cutoff * index * T_) /
                    (TYPE(2) * M_PI * cutoff * index * T_);
        }
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
size_t internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::size() const {
    return this->sinc.size();
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void internal::Sinc<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::applyWindow(
    const Window::Window<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2>&
        window) {
    for(int i = 0; i < (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR; ++i)
        sinc[i] *= window[i + (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR];
}

// =============================================================================
//  IMPLEMENTATIONS: Oversampler Lifecycle
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::Oversampler() {}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::configure(
    TYPE sampleRate,
    int maxChannels,
    int maxBlockSize) {
    if(sampleRate <= TYPE(0.0) || maxChannels <= 0 || maxBlockSize <= 0)
        throw std::invalid_argument("Invalid configuration.");

    sinc.configure(static_cast<TYPE>(sampleRate));
    sinc.applyWindow(
        Window::Kaiser<TYPE, (SINC_RADIUS + 1) * OVERSAMPLE_FACTOR * 2> {5.0});

    int maxInterpolatedSize = maxBlockSize * OVERSAMPLE_FACTOR;
    beginBuf.clear();
    endBuf.clear();
    decBeginBuf.clear();
    decEndBuf.clear();
    beginBuf.resize(maxChannels, CircularBuffer<TYPE, SINC_RADIUS>(TYPE(0.0)));
    endBuf.resize(maxChannels, CircularBuffer<TYPE, SINC_RADIUS>(TYPE(0.0)));
    decBeginBuf.resize(
        maxChannels,
        CircularBuffer<TYPE, SINC_RADIUS * OVERSAMPLE_FACTOR>(TYPE(0.0)));
    decEndBuf.resize(
        maxChannels,
        CircularBuffer<TYPE, SINC_RADIUS * OVERSAMPLE_FACTOR>(TYPE(0.0)));

    x_interp.clear();
    x_decim.clear();
    x_interp.resize(maxChannels, std::vector<TYPE>(maxBlockSize + SINC_RADIUS));
    x_decim.resize(maxChannels,
                   std::vector<TYPE>(maxInterpolatedSize +
                                     SINC_RADIUS * OVERSAMPLE_FACTOR));
    interpolatedBuf.resize(maxChannels, std::vector<TYPE>(maxInterpolatedSize));
}

// =============================================================================
//  IMPLEMENTATIONS: Interpolate
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<resinc_traits::is_juce_type<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(T&& buffer) {
    interpolate_helper(buffer.getArrayOfReadPointers(),
                       buffer.getNumChannels(),
                       buffer.getNumSamples());
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(T&& buffer) {
    const TYPE* ptr = buffer.data();
    interpolate_helper(&ptr, 1, static_cast<int>(buffer.size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(T&& buffer) {
    int channels = static_cast<int>(buffer.size());
    if(channels == 0) return;
    std::vector<const TYPE*> ptrs(channels);
    for(int i = 0; i < channels; ++i) ptrs[i] = buffer[i].data();
    interpolate_helper(
        ptrs.data(), channels, static_cast<int>(buffer[0].size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate(
    const TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    interpolate_helper(ptrToBuffers, numChannels, numSamples);
}

// =============================================================================
//  IMPLEMENTATIONS: Decimate
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<resinc_traits::is_juce_type<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(T& buffer) {
    decimate_helper(buffer.getArrayOfWritePointers(),
                    buffer.getNumChannels(),
                    buffer.getNumSamples());
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(T& buffer) {
    TYPE* ptr = buffer.data();
    decimate_helper(&ptr, 1, static_cast<int>(buffer.size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(T& buffer) {
    int channels = static_cast<int>(buffer.size());
    if(channels == 0) return;
    std::vector<TYPE*> ptrs(channels);
    for(int i = 0; i < channels; ++i) ptrs[i] = buffer[i].data();
    decimate_helper(ptrs.data(), channels, static_cast<int>(buffer[0].size()));
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate(
    TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    decimate_helper(ptrToBuffers, numChannels, numSamples);
}

// =============================================================================
//  IMPLEMENTATIONS: Process
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename T, typename Callable>
typename std::enable_if_t<resinc_traits::is_juce_type<std::decay_t<T>>::value>
    Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::process(
        Callable&& processBlock) {
    const int ch = static_cast<int>(interpolatedBuf.size());
    const int len = static_cast<int>(interpolatedBuf[0].size());
    std::vector<TYPE*> ptrs(ch);
    for(int i = 0; i < ch; ++i) ptrs[i] = interpolatedBuf[i].data();
    T buffer(static_cast<TYPE* const*>(ptrs.data()), ch, len);
    processBlock(buffer);
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename Callable>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::process(
    Callable&& processBlock) {
    processBlock(interpolatedBuf);
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
template<typename Callable>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::processEach(
    Callable&& processBlock) {
    for(auto& i : interpolatedBuf) {
        processBlock(i);
    }
}

// =============================================================================
//  IMPLEMENTATIONS: Helpers
// =============================================================================

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::interpolate_helper(
    const TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    if(numChannels <= 0 || numSamples <= 0) {
        throw std::invalid_argument(
            "Number of channels and samples must be positive.");
    }

    if(static_cast<size_t>(numChannels) > x_interp.size() ||
       static_cast<size_t>(numSamples + SINC_RADIUS) > x_interp[0].size())
        throw std::runtime_error(
            "Oversampler: Input size exceeds configured maxBlockSize.");

    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0; i < numSamples + SINC_RADIUS; ++i) {
            x_interp[channel][i] =
                (i >= SINC_RADIUS ? ptrToBuffers[channel][i - SINC_RADIUS] :
                                    endBuf[channel][i]);
        }
    }

    int interpolatedBufSize = numSamples * OVERSAMPLE_FACTOR;

    for(int channel = 0; channel < numChannels; channel++) {
        interpolatedBuf[channel][0] = x_interp[channel][0];
        for(int k = 1; k < interpolatedBufSize; k++) {
            int delta = k % OVERSAMPLE_FACTOR;
            int index = k / OVERSAMPLE_FACTOR;
            if(delta != 0) {
                interpolatedBuf[channel][k] = 0;
                for(int n = -SINC_RADIUS; n <= 0; n++)
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) * x_interp[channel][index - n];
                for(int n = 1; n <= SINC_RADIUS; n++)
                    interpolatedBuf[channel][k] +=
                        this->sinc(n, delta) *
                        beginBuf[channel][SINC_RADIUS - n];
            } else {
                interpolatedBuf[channel][k] = x_interp[channel][index];
                beginBuf[channel].push(x_interp[channel][index - 1]);
            }
        }
        beginBuf[channel].push(x_interp[channel][numSamples - 1]);
        for(int i = 0; i < SINC_RADIUS; i++)
            endBuf[channel].push(x_interp[channel][numSamples + i]);
    }
}

template<typename TYPE, int OVERSAMPLE_FACTOR, int SINC_RADIUS>
void Oversampler<TYPE, OVERSAMPLE_FACTOR, SINC_RADIUS>::decimate_helper(
    TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    if(static_cast<size_t>(numChannels) > x_decim.size() ||
       static_cast<size_t>(numSamples) >
           (interpolatedBuf[0].size() / OVERSAMPLE_FACTOR))
        throw std::runtime_error(
            "Oversampler: Output size exceeds configured maxBlockSize.");

    int interpolatedBufSize = static_cast<int>(interpolatedBuf[0].size());
    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0;
            i < interpolatedBufSize + SINC_RADIUS * OVERSAMPLE_FACTOR;
            ++i) {
            x_decim[channel][i] =
                (i >= SINC_RADIUS * OVERSAMPLE_FACTOR ?
                     interpolatedBuf[channel]
                                    [i - SINC_RADIUS * OVERSAMPLE_FACTOR] :
                     decEndBuf[channel][i]);
        }
    }

    for(int channel = 0; channel < numChannels; channel++) {
        for(int k = 0; k < numSamples; k++) {
            int index = OVERSAMPLE_FACTOR * k;
            ptrToBuffers[channel][k] = 0;
            for(int n = 1; n <= SINC_RADIUS * OVERSAMPLE_FACTOR; n++)
                ptrToBuffers[channel][k] +=
                    sinc[n] *
                    decBeginBuf[channel][SINC_RADIUS * OVERSAMPLE_FACTOR - n];
            for(int n = 0; n >= -(SINC_RADIUS * OVERSAMPLE_FACTOR); n--)
                ptrToBuffers[channel][k] +=
                    sinc[n] * x_decim[channel][index - n];
            for(int i = 0; i < OVERSAMPLE_FACTOR; i++)
                decBeginBuf[channel].push(x_decim[channel][index + i]);
            ptrToBuffers[channel][k] /= OVERSAMPLE_FACTOR;
        }
        for(int i = 0; i < SINC_RADIUS * OVERSAMPLE_FACTOR; i++)
            decEndBuf[channel].push(x_decim[channel][interpolatedBufSize + i]);
    }
}

// =============================================================================
//  IMPLEMENTATIONS: Resampler Lifecycle
// =============================================================================
template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
Resampler<TYPE, SINC_RADIUS, RESOLUTION>::Resampler() {}

template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
void Resampler<TYPE, SINC_RADIUS, RESOLUTION>::configure(TYPE sampleRate,
                                                         TYPE targetSampleRate,
                                                         int maxChannels,
                                                         int maxBlockSize) {
    if(sampleRate <= TYPE(0.0) || targetSampleRate <= TYPE(0.0) ||
       maxChannels <= 0 || maxBlockSize <= 0)
        throw std::invalid_argument("Invalid configuration.");

    TYPE cutoff = std::min(sampleRate, targetSampleRate) / 2.0;
    sinc.configure_with_cutoff(cutoff, sampleRate);
    sinc.applyWindow(
        Window::Kaiser<TYPE, (SINC_RADIUS + 1) * RESOLUTION * 2> {5.0});

    current_phase = TYPE(0.0);
    maxOutputSize =
        maxBlockSize * static_cast<int>(targetSampleRate / sampleRate + 1);
    oversampleFactor = targetSampleRate / sampleRate;
    beginBuf.clear();
    endBuf.clear();
    beginBuf.resize(maxChannels, CircularBuffer<TYPE, SINC_RADIUS>(TYPE(0.0)));
    endBuf.resize(maxChannels, CircularBuffer<TYPE, SINC_RADIUS>(TYPE(0.0)));

    x.clear();
    x.resize(maxChannels, std::vector<TYPE>(maxBlockSize + SINC_RADIUS + 1));
}

// =============================================================================
//  IMPLEMENTATIONS: Resample
// =============================================================================
template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
template<typename T1, typename T2>
typename std::enable_if_t<
    resinc_traits::is_juce_type<std::decay_t<T1>>::value &&
        resinc_traits::is_juce_type<std::decay_t<T2>>::value,
    int>
    Resampler<TYPE, SINC_RADIUS, RESOLUTION>::resample(T1&& input, T2& output) {
    return resample_helper(input.getArrayOfReadPointers(),
                           output.getArrayOfWritePointers(),
                           input.getNumChannels(),
                           output.getNumSamples());
}

template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
template<typename T1, typename T2>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T1>>::value &&
        resinc_traits::is_single_channel<std::decay_t<T2>>::value,
    int>
    Resampler<TYPE, SINC_RADIUS, RESOLUTION>::resample(T1&& input, T2& output) {
    const TYPE* ptr = input.data();
    TYPE* outPtr = output.data();
    return resample_helper(&ptr, &outPtr, 1, static_cast<int>(input.size()));
}

template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
template<typename T1, typename T2>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T1>>::value &&
        resinc_traits::is_multi_channel<std::decay_t<T2>>::value,
    int>
    Resampler<TYPE, SINC_RADIUS, RESOLUTION>::resample(T1&& input, T2& output) {
    int channels = static_cast<int>(input.size());
    if(channels == 0) return 0;
    std::vector<const TYPE*> inPtrs(channels);
    std::vector<TYPE*> outPtrs(channels);
    for(int i = 0; i < channels; i++) {
        inPtrs[i] = input[i].data();
        outPtrs[i] = output[i].data();
    }
    return resample_helper(inPtrs.data(),
                           outPtrs.data(),
                           channels,
                           static_cast<int>(input[0].size()));
}

template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
int Resampler<TYPE, SINC_RADIUS, RESOLUTION>::resample(
    const TYPE* const* ptrToInBuffers,
    TYPE* const* ptrToBuffers,
    int numChannels,
    int numSamples) {
    return resample_helper(
        ptrToInBuffers, ptrToBuffers, numChannels, numSamples);
}

// =============================================================================
//  IMPLEMENTATIONS: Helpers
// =============================================================================
template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
int Resampler<TYPE, SINC_RADIUS, RESOLUTION>::resample_helper(
    const TYPE* const* ptrToInBuffers,
    TYPE* const* ptrToOutBuffers,
    int numChannels,
    int numSamples) {
    if(numChannels <= 0 || numSamples <= 0) {
        throw std::invalid_argument(
            "Number of channels and samples must be positive.");
    }

    if(static_cast<size_t>(numChannels) > x.size() ||
       static_cast<size_t>(numSamples + SINC_RADIUS) > x[0].size())
        throw std::runtime_error(
            "Resampler: Input size exceeds configured maxBlockSize.");

    for(int channel = 0; channel < numChannels; channel++) {
        for(int i = 0; i < numSamples + SINC_RADIUS; ++i) {
            x[channel][i] =
                (i >= SINC_RADIUS ? ptrToInBuffers[channel][i - SINC_RADIUS] :
                                    endBuf[channel][i]);
        }
    }

    int outputCount =
        static_cast<int>((numSamples - current_phase) * oversampleFactor);

    TYPE gainScale = (oversampleFactor < 1.0) ?
                         static_cast<TYPE>(oversampleFactor) :
                         TYPE(1.0);
    int delta;
    for(int channel = 0; channel < numChannels; channel++) {
        for(int k = 0; k < outputCount; ++k) {
            int index = static_cast<int>(
                (static_cast<double>(k) / oversampleFactor) + current_phase);
            delta =
                static_cast<int>(((static_cast<double>(k) / oversampleFactor) +
                                  current_phase - index) *
                                 RESOLUTION);
            if(delta != 0) {
                ptrToOutBuffers[channel][k] = 0;
                for(int n = -SINC_RADIUS; n <= 0; n++)
                    ptrToOutBuffers[channel][k] +=
                        this->sinc(n, delta) * x[channel][index - n];
                for(int n = 1; n <= SINC_RADIUS; n++)
                    ptrToOutBuffers[channel][k] +=
                        this->sinc(n, delta) *
                        beginBuf[channel][SINC_RADIUS - n];
            } else {
                ptrToOutBuffers[channel][k] = x[channel][index];

                if(index > 0) {
                    beginBuf[channel].push(x[channel][index - 1]);
                } else {
                    beginBuf[channel].push(beginBuf[channel][SINC_RADIUS - 1]);
                }
            }
            ptrToOutBuffers[channel][k] *= gainScale;
        }
        beginBuf[channel].push(x[channel][numSamples - 1]);
        for(int i = 0; i < SINC_RADIUS; i++)
            endBuf[channel].push(x[channel][numSamples + i]);
    }
    current_phase =
        (current_phase + (outputCount / oversampleFactor)) - numSamples;
    return outputCount;
}

// =============================================================================
//  IMPLEMENTATIONS: resample
// =============================================================================
template<typename TYPE,
         int SINC_RADIUS,
         int RESOLUTION,
         typename T1,
         typename T2>
typename std::enable_if_t<
    resinc_traits::is_juce_type<std::decay_t<T1>>::value &&
        resinc_traits::is_juce_type<std::decay_t<T2>>::value,
    int>
    resample(T1&& input,
             T2& output,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate) {
    return internal::resample_helper<TYPE, SINC_RADIUS, RESOLUTION>(
        input.getArrayOfReadPointers(),
        output.getArrayOfWritePointers(),
        input.getNumChannels(),
        output.getNumSamples(),
        sourceSampleRate,
        targetSampleRate);
}

template<typename TYPE,
         int SINC_RADIUS,
         int RESOLUTION,
         typename T1,
         typename T2>
typename std::enable_if_t<
    resinc_traits::is_single_channel<std::decay_t<T1>>::value &&
        resinc_traits::is_single_channel<std::decay_t<T2>>::value,
    int>
    resample(T1&& input,
             T2& output,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate) {
    const TYPE* ptr = input.data();
    TYPE* outPtr = output.data();
    return internal::resample_helper<TYPE, SINC_RADIUS, RESOLUTION>(
        &ptr,
        &outPtr,
        1,
        static_cast<int>(input.size()),
        sourceSampleRate,
        targetSampleRate);
}

template<typename TYPE,
         int SINC_RADIUS,
         int RESOLUTION,
         typename T1,
         typename T2>
typename std::enable_if_t<
    resinc_traits::is_multi_channel<std::decay_t<T1>>::value &&
        resinc_traits::is_multi_channel<std::decay_t<T2>>::value,
    int>
    resample(T1&& input,
             T2& output,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate) {
    int channels = static_cast<int>(input.size());
    if(channels == 0) return 0;
    std::vector<const TYPE*> inPtrs(channels);
    std::vector<TYPE*> outPtrs(channels);
    for(int i = 0; i < channels; i++) {
        inPtrs[i] = input[i].data();
        outPtrs[i] = output[i].data();
    }
    return internal::resample_helper<TYPE, SINC_RADIUS, RESOLUTION>(
        inPtrs.data(),
        outPtrs.data(),
        channels,
        static_cast<int>(input[0].size()),
        sourceSampleRate,
        targetSampleRate);
}

template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
int resample(const TYPE* const* ptrToInBuffers,
             TYPE* const* ptrToOutBuffers,
             int numChannels,
             int numSamples,
             non_deduced<TYPE> sourceSampleRate,
             non_deduced<TYPE> targetSampleRate) {
    return internal::resample_helper<TYPE, SINC_RADIUS, RESOLUTION>(
        ptrToInBuffers,
        ptrToOutBuffers,
        numChannels,
        numSamples,
        sourceSampleRate,
        targetSampleRate);
}

// =============================================================================
//  IMPLEMENTATIONS: Helpers
// =============================================================================

template<typename TYPE, int SINC_RADIUS, int RESOLUTION>
int internal::resample_helper(const TYPE* const* ptrToInBuffers,
                              TYPE* const* ptrToOutBuffers,
                              int numChannels,
                              int numSamples,
                              TYPE sourceSampleRate,
                              TYPE targetSampleRate) {
    if(numChannels <= 0 || numSamples <= 0) return 0;

    internal::Sinc<TYPE, RESOLUTION, SINC_RADIUS> sinc;
    TYPE cutoff = std::min(sourceSampleRate, targetSampleRate) / TYPE(2.0);
    sinc.configure_with_cutoff(cutoff, sourceSampleRate);
    sinc.applyWindow(
        Window::Kaiser<TYPE, (SINC_RADIUS + 1) * RESOLUTION * 2> {5.0});

    const TYPE invOversampleFactor = sourceSampleRate / targetSampleRate;
    const TYPE oversampleFactor = targetSampleRate / sourceSampleRate;
    const int outputCount = static_cast<int>(numSamples * oversampleFactor);
    const TYPE gainScale = (oversampleFactor < TYPE(1.0)) ?
                               static_cast<TYPE>(oversampleFactor) :
                               TYPE(1.0);

    for(int k = 0; k < outputCount; ++k) {
        const double virtualIndex =
            static_cast<double>(k) * invOversampleFactor;
        const int index = static_cast<int>(virtualIndex);
        const int delta = static_cast<int>((virtualIndex - index) * RESOLUTION);

        // --- Skip Zeros: Calculate valid n-range ---
        // Constraint 1: n >= -SINC_RADIUS && n <= SINC_RADIUS
        // Constraint 2: 0 <= (index - n) < numSamples
        //   => n <= index
        //   => n > index - numSamples
        const int n_start = std::max(-SINC_RADIUS, index - numSamples + 1);
        const int n_end = std::min(SINC_RADIUS, index);

        for(int channel = 0; channel < numChannels; channel++) {
            TYPE acc = 0;
            const TYPE* in = ptrToInBuffers[channel];

            // Hot loop: No branches, minimal iterations at boundaries
            for(int n = n_start; n <= n_end; n++) {
                acc += sinc(n, delta) * in[index - n];
            }

            ptrToOutBuffers[channel][k] = acc * gainScale;
        }
    }
    return outputCount;
}