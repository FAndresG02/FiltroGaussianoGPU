// Librería para cargar imágenes
#define STB_IMAGE_IMPLEMENTATION 
#include "C:\\Users\\andre\\Documents\\COMPUTACION_PARALELA\\Librerias\\stb_image.h" 
// Librería para guardar imágenes
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "C:\\Users\\andre\\Documents\\COMPUTACION_PARALELA\\Librerias\\stb_image_write.h"  

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "C:\\Users\\andre\\Documents\\COMPUTACION_PARALELA\\Book\\book.h"  
#include "cuda_runtime.h"  // Librería CUDA para gestionar la GPU

#ifndef M_PI
#define M_PI 3.14159265358979323846  // Definir PI 
#endif

#define KERNEL_SIZE 9  // Tamaño del kernel 
#define OFFSET (KERNEL_SIZE / 2)  // Offset necesario para acceder a la vecindad del píxel

// Kernel gaussiano en memoria constante de la GPU
__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];  

// Generar kernel gaussiano en el host
void generarKernelGaussiano(float* kernel, int kernelSize, float sigma) {
    float sum = 0.0f;
    int offset = kernelSize / 2;

    // Llenar el kernel con valores de la distribución gaussiana
    for (int y = -offset; y <= offset; y++) {
        for (int x = -offset; x <= offset; x++) {
            // Fórmula de la gaussiana
            float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
            float value = expf(exponent) / (2.0f * M_PI * sigma * sigma);
            kernel[(y + offset) * kernelSize + (x + offset)] = value;
            sum += value;  // Acumular la suma total para la normalización
        }
    }

    // Normalizar kernel para que la suma total sea 1
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
}

// CUDA kernel para aplicar filtro gaussiano
__global__ void filtroGaussianoKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    // Comprobar si el hilo está dentro de los límites de la imagen
    if (x >= width || y >= height) return;

    float sum = 0.0f;

    // Recorrer el vecindario del píxel para aplicar el filtro gaussiano
    for (int ky = -OFFSET; ky <= OFFSET; ky++) {
        for (int kx = -OFFSET; kx <= OFFSET; kx++) {
            int px = min(max(x + kx, 0), width - 1);  
            int py = min(max(y + ky, 0), height - 1);  
            float pixel = static_cast<float>(input[py * width + px]); 
            sum += pixel * d_kernel[(ky + OFFSET) * KERNEL_SIZE + (kx + OFFSET)];  
        }
    }

    // Asignar el valor resultante al píxel de salida, limitándolo entre 0 y 255
    output[y * width + x] = min(max(int(sum), 0), 255);
}

int main() {
    int width, height, channels;

    // Cargar la imagen en escala de grises
    unsigned char* gray = stbi_load("C:/Users/andre/Documents/TRABAJO/Tarea en clase - Filtro/img.jpg", &width, &height, &channels, 1);
    if (!gray) {
        printf("No se pudo cargar la imagen.\n");
        return -1;
    }

    size_t imageSize = width * height * sizeof(unsigned char);  // Tamaño de la imagen en memoria
    unsigned char* result = (unsigned char*)malloc(imageSize);  // Reservar memoria para la imagen de salida
    HANDLE_NULL(result);  // Comprobar si la asignación de memoria fue exitosa

    // Generar el kernel gaussiano en el host
    float h_kernel[KERNEL_SIZE * KERNEL_SIZE];  
    float sigma = 1.5f;  // Valor de sigma para la distribución gaussiana
    generarKernelGaussiano(h_kernel, KERNEL_SIZE, sigma); 

    // Copiar el kernel a memoria constante en la GPU
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE)); 

    unsigned char* d_input, * d_output;
    HANDLE_ERROR(cudaMalloc((void**)&d_input, imageSize));  // Reservar memoria en la GPU para la entrada
    HANDLE_ERROR(cudaMalloc((void**)&d_output, imageSize));  // Reservar memoria en la GPU para la salida
    HANDLE_ERROR(cudaMemcpy(d_input, gray, imageSize, cudaMemcpyHostToDevice));  // Copiar la imagen a la GPU

    // Inicializar eventos de CUDA para medir el tiempo de ejecución
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);   
    cudaEventRecord(start, 0);  

    // Configuración de los hilos y bloques para la ejecución en la GPU
    dim3 threads(32, 32);  // Configuración de 32x32 hilos por bloque
    dim3 blocks((width + 31) / 32, (height + 31) / 32);  // Número de bloques en función del tamaño de la imagen

    // Lanzamiento del kernel en la GPU para aplicar el filtro gaussiano
    filtroGaussianoKernel << <blocks, threads >> > (d_input, d_output, width, height);

    // Medición del tiempo de ejecución
    cudaEventRecord(stop, 0);  
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    // Copiar la imagen de vuelta a la memoria del host
    HANDLE_ERROR(cudaMemcpy(result, d_output, imageSize, cudaMemcpyDeviceToHost));  

    // Guardar la imagen resultante
    if (stbi_write_jpg("C:/Users/andre/Documents/9Cuda.jpg", width, height, 1, result, 100)) {
        printf("Imagen JPG guardada correctamente.\n");
    }
    else {
        printf("Error al guardar la imagen JPG.\n");
    }

    printf("Filtro gaussiano aplicado en GPU.\n");
    // Mostrar el tiempo de ejecución
    printf("Tiempo total de ejecución: %.2f ms\n", elapsedTime); 

    // Liberar recursos
    stbi_image_free(gray);  
    free(result);  
    cudaFree(d_input);  
    cudaFree(d_output);  
    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 

    return 0;
}
