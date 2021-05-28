// Copyright 2021 Solovev Aleksandr
#include "../../../modules/task_4/solovev_a_radix_sort/radix_sort.h"

#include "../../../3rdparty/unapproved/unapproved.h"

int generateRandomArray(std::vector<int> *buffer, int min, int max) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    for (size_t i = 0; i < buffer->size(); i++) {
        buffer->at(i) = static_cast<int>(gen()) % (max - min + 1) + min;
    }
    return 0;
}

int MergeArrays(std::vector<int> *buffer1, std::vector<int> *buffer2, int left, std::vector<int> *result) {
    int current_buff = 0;
    for (size_t i = left; i < buffer1->size() + left; i++) {
        result->at(i) = buffer1->at(current_buff);
        current_buff++;
    }
    current_buff = 0;
    for (size_t i = left + buffer1->size() ; i < buffer2->size() + left + buffer1->size(); i++) {
        result->at(i) = buffer2->at(current_buff);
        current_buff++;
    }

    return 0;
}

int SortingCheck(std::vector<int> *buffer) {
    for (size_t i = 0; i < buffer->size() - 1; i++) {
        if (buffer->at(i) <= buffer->at(i + 1)) {
            continue;
        } else {
            return -1;
        }
    }
    return 0;
}

void CountingSort(std::vector<int> *input, std::vector<int> *output, int valbyte) {
    unsigned char *buffer = (unsigned char *)input->data();
    int counter[256] = {0};
    size_t length_counter = 256;
    int value = 0;

    for (size_t i = 0; i < input->size(); i++) {
        counter[buffer[4 * i + valbyte]]++;
    }

    for (size_t j = 0; j < length_counter; j++) {
        int tmp = counter[j];
        counter[j] = value;
        value += tmp;
    }

    for (size_t i = 0; i < input->size(); i++) {
        output->at(counter[buffer[4 * i + valbyte]]++) = input->at(i);
    }
}

int RadixSortUnsigned(std::vector<int> *buffer) {
    if (buffer->empty()) {
        return -1;
    }
    std::vector<int> outbuf(buffer->size());
    if (outbuf.data() == nullptr) {
        return -1;
    }
    for (int i = 0; i < 2; i++) {
        CountingSort(buffer, &outbuf, 2 * i);
        CountingSort(&outbuf, buffer, 2 * i + 1);
    }
    return 0;
}
int RadixSortUnsignedParallel(std::vector<int> *buffer) {
    if (buffer->empty()) {
        return -1;
    }
    std::vector<int> outbuf(buffer->size());
    if (outbuf.data() == nullptr) {
        return -1;
    }
    for (int i = 0; i < 2; i++) {
        CountingSort(buffer, &outbuf, 2 * i);
        CountingSort(&outbuf, buffer, 2 * i + 1);
    }
    return 0;
}
int RadixSort(std::vector<int> *buffer) {
    if (buffer->size() < 1)
        return -1;
    int positive_length = 0, negative_length = 0;
    int status = 0;

    for (size_t i = 0; i < buffer->size(); i++) {
        if (buffer->at(i) >= 0) {
            positive_length++;
        } else {
            negative_length++;
        }
    }
    std::vector<int> positive_numbers(positive_length);
    std::vector<int> negative_numbers(negative_length);
    positive_length = 0;
    negative_length = 0;
    for (size_t i = 0; i < buffer->size(); i++) {
        if (buffer->at(i) >= 0) {
            positive_numbers[positive_length] = buffer->at(i);
            positive_length++;
        } else {
            negative_numbers[negative_length] = buffer->at(i);
            negative_length++;
        }
    }

    status = RadixSortUnsigned(&positive_numbers);
    status = RadixSortUnsigned(&negative_numbers);

    status = MergeArrays(&negative_numbers,  &positive_numbers, 0,  buffer);
    return status;
}
int RadixSortParallel(std::vector<int> *buffer, int left, int right) {
    if (buffer->size() < 1)
        return -1;
    int positive_length = 0, negative_length = 0;
    int status = 0;
    for (int i = left; i <= right; i++) {
        if (buffer->at(i) >= 0) {
            positive_length+=1;
        } else {
            negative_length+=1;
        }
    }
    std::vector<int> positive_numbers(positive_length);
    std::vector<int> negative_numbers(negative_length);
    positive_length = 0;
    negative_length = 0;

    for (int i = left; i <= right; i++) {
        if (buffer->at(i) >= 0) {
            positive_numbers[positive_length] = buffer->at(i);
            positive_length+=1;
        } else {
            negative_numbers[negative_length] = buffer->at(i);
            negative_length+=1;
        }
    }

    status = RadixSortUnsigned(&positive_numbers);
    status = RadixSortUnsigned(&negative_numbers);

    status = MergeArrays(&negative_numbers,  &positive_numbers, left,  buffer);

    return status;
}
void merge(int* a, int size_a, int* b, int size_b) {
    int i = 0, j = 0, k = 0;
    int size_c = size_a + size_b;
    int* c = new int[size_c];
    while ((i < size_a) && (j < size_b)) {
        if (a[i] <= b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }

    if (i == size_a) {
        while (j < size_b) {
            c[k++] = b[j++];
        }
    } else {
        while (i < size_a) {
            c[k++] = a[i++];
        }
    }

    for (int i = 0; i < size_c; i++) {
        a[i] = c[i];
    }
    delete[] c;
}

int ParallelSortingSTD(std::vector<int> *arr, int thread_count) {
    std::vector<std::thread> threads;
    int part_vec_size = static_cast<int>(arr->size()) / thread_count;
    for (int i = 0; i < thread_count; i++) {
        std::thread th([&arr, &thread_count, &part_vec_size] (int i) {
            if (i != thread_count - 1) {
                RadixSortParallel(arr, i * part_vec_size, (i + 1) * part_vec_size - 1);
            } else {
                RadixSortParallel(arr, i * part_vec_size, static_cast<int>(arr->size()) - 1);
            }
        }, i);
        threads.emplace_back(std::move(th));
    }

    for (int i = 0; i < thread_count; i++)
        threads[i].join();
    for (int i = 1; i < thread_count; i++) {
        int current_size = part_vec_size;
        if (i == thread_count - 1) {
            current_size = static_cast<int>(arr->size()) - (thread_count - 1) * part_vec_size;
        }
        merge(arr->data(), part_vec_size * i, arr->data() + part_vec_size * i, current_size);
    }
    return 0;
}
