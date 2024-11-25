#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

__global__ void normalize(PPMPixel *deviceData, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    deviceData[i].red = (deviceData[i].red * 4) / 256;
    deviceData[i].blue = (deviceData[i].blue * 4) / 256;
    deviceData[i].green = (deviceData[i].green * 4) / 256;
  }
}

__global__ void countPatternsInArray (PPMPixel * deviceData, int nThreads, int n, float * counts) {
  int i = blockIdx.z * nThreads + threadIdx.x;
  if (i < n) 
    if (deviceData[i].red == blockIdx.x && deviceData[i].green == blockIdx.y/4 && deviceData[i].blue == blockIdx.y%4) 
      atomicAdd(&counts[blockIdx.x * 16 + blockIdx.y], 1);
}

__global__ void division (float * counts, float n) {
  counts[threadIdx.x] /= n;
}


double Histogram(PPMImage *image, float *h) {
  float ms;
  cudaEvent_t start, stop;
  // Create Events
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  // Launch kernel and compute kernel runtime.
  // Warning: make sure only the kernel is being profiled, memcpies should be
  // out of this region.
  // vars
  int rows, cols, nBlocks, nThreads, n, i;
  PPMPixel * devicePixels;
  float * deviceCounts;
  unsigned int bytes;
  //set values
  cols = image->y;
  rows = image->x;
  n = cols * rows;
  bytes = sizeof(PPMPixel) * n;
  nThreads = 512;
  nBlocks = (n + 511)/nThreads;
  dim3 grid(4, 16, nBlocks);
  dim3 block (nThreads);
  CUDACHECK(cudaEventRecord(start));
	//create memory
  CUDACHECK(cudaMalloc(&devicePixels, bytes));
  CUDACHECK(cudaMemcpy(devicePixels, image->data, bytes, cudaMemcpyHostToDevice));
  bytes = sizeof(float) * 64;
  CUDACHECK(cudaMalloc(&deviceCounts, bytes));
  CUDACHECK(cudaMemset(deviceCounts, 0, bytes));
  //normalize
  normalize<<<nBlocks, nThreads>>>(devicePixels, n);
  CUDACHECK(cudaDeviceSynchronize());
  //count
  countPatternsInArray<<<grid, block>>> (devicePixels, nThreads, n, deviceCounts);
  CUDACHECK(cudaDeviceSynchronize());
  //divide
  division<<<1, 64>>> (deviceCounts, float(n));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(h, deviceCounts, bytes, cudaMemcpyDeviceToHost));
  //free memory
  CUDACHECK(cudaFree(devicePixels));
  CUDACHECK(cudaFree(deviceCounts));
  CUDACHECK(cudaEventRecord(stop));
/*
  for (i = 0; i < n; i++) {
    image->data[i].red = floor((image->data[i].red * 4) / 256);
    image->data[i].blue = floor((image->data[i].blue * 4) / 256);
    image->data[i].green = floor((image->data[i].green * 4) / 256);
  }
  memset(counts, 0, bytes);
	printf("%d %d %d\n", nBlocks, nThreads, n);
  int blkX, blkY, tX;
  for (blkX = 0; blkX < 64; ++blkX)
    for (blkY = 0; blkY < nBlocks; ++blkY)
      for (tX = 0; tX < nThreads; ++tX) {
        int i = blkY * nThreads + tX,
            j = blkX/16,
            k = (blkX - j * 16)/4,
            l = blkX%4;
        if (i < n && j < 4 && k < 4 && l < 4) {
          if (image->data[i].red == j && image->data[i].green == k && image->data[i].blue == l) 
            ++counts[blkX];
	} else {
		printf("ola\n");
	}
      }
*/
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&ms, start, stop));
  // Destroy events
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));

  return ((double)ms) / 1000.0;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  PPMImage *image = readPPM(argv[1]);
  float *h = (float *)malloc(sizeof(float) * 64);

  // Initialize histogram
  for (int i = 0; i < 64; i++)
    h[i] = 0.0;

  // Compute histogram
  double t = Histogram(image, h);

  for (int i = 0; i < 64; i++)
    printf("%0.3f ", h[i]);
  printf("\n");

  fprintf(stderr, "%lf\n", t);
  free(h);
}
