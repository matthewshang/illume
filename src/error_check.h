#ifndef _ERROR_CHECK_
#define _ERROR_CHECK_

static void handle_error(cudaError_t err, const char *file, int line) 
{
    if (err != cudaSuccess) 
    {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

#endif