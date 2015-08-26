#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mmintrin.h>
#include <emmintrin.h>

#define M_PI 3.1415926535f

typedef struct {
        float x, y;
} Vec2;

static inline Vec2 random_gradient()
{
	const float v = (float)rand() / RAND_MAX * M_PI * 2.0f;
	return (Vec2){cosf(v), sinf(v)};
}

typedef struct {
        Vec2 rgradients[512];
        int  permutations[512];
} Noise2DContext;

static __attribute__((noinline)) float noise2d_get(Noise2DContext *ctx, float x, float y)
{
	float x0f = floorf(x);
	float y0f = floorf(y);
	int x0 = x0f;
	int y0 = y0f;
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	int idx1 = ctx->permutations[x0 & 255] + ctx->permutations[y0 & 255];
	int idx2 = ctx->permutations[x1 & 255] + ctx->permutations[y0 & 255];
	int idx3 = ctx->permutations[x0 & 255] + ctx->permutations[y1 & 255];
	int idx4 = ctx->permutations[x1 & 255] + ctx->permutations[y1 & 255];

    idx1 &= 255;
    idx2 &= 255;
    idx3 &= 255;
    idx4 &= 255;

    __m128 rgradientsx = _mm_setr_ps(ctx->rgradients[idx1].x, ctx->rgradients[idx2].x, ctx->rgradients[idx3].x, ctx->rgradients[idx4].x);
    __m128 rgradientsy = _mm_setr_ps(ctx->rgradients[idx1].y, ctx->rgradients[idx2].y, ctx->rgradients[idx3].y, ctx->rgradients[idx4].y);
    __m128 xsplat = _mm_set1_ps(x);
    __m128 ysplat = _mm_set1_ps(y);
    __m128 x0fsplat = _mm_set1_ps(x0f);
    __m128 y0fsplat = _mm_set1_ps(y0f);
    __m128 xv = _mm_mul_ps(rgradientsx, _mm_sub_ps(xsplat, _mm_add_ps(x0fsplat, _mm_setr_ps(0.0f, 1.0f, 0.0f, 1.0f))));
    __m128 yv = _mm_mul_ps(rgradientsy, _mm_sub_ps(ysplat, _mm_add_ps(y0fsplat, _mm_setr_ps(0.0f, 0.0f, 1.0f, 1.0f))));
    __m128 v = _mm_add_ps(xv, yv);

     union {
        __m128 v;    
        float a[4];  
    } converter;
    converter.v = v;
         
    float xf = x - x0f;
    float yf = y - y0f;
    
    float v0 = ctx->rgradients[idx1].x * (xf - 0.0f) + ctx->rgradients[idx1].y * (yf - 0.0f);
    float v1 = ctx->rgradients[idx2].x * (xf - 1.0f) + ctx->rgradients[idx2].y * (yf - 0.0f);
    float v2 = ctx->rgradients[idx3].x * (xf - 0.0f) + ctx->rgradients[idx3].y * (yf - 1.0f);
    float v3 = ctx->rgradients[idx4].x * (xf - 1.0f) + ctx->rgradients[idx4].y * (yf - 1.0f);
	       /*
    float v0 = converter.a[0];
    float v1 = converter.a[1];
    float v2 = converter.a[2];
    float v3 = converter.a[3];
           */
    float fx = xf * xf * (3.0f - 2 * xf);
    float fy = yf * yf * (3.0f - 2 * yf);
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;
    float vx0 = v0 * fx1 + v1 * fx;
    float vx1 = v2 * fx1 + v3 * fx;
    return vx0 * fy1 + vx1 * fy;
}

static void init_noise2d(Noise2DContext *ctx)
{
        for (int i = 0; i < 256; i++)
                ctx->rgradients[i + 256] = ctx->rgradients[i] = random_gradient();

        for (int i = 0; i < 256; i++) {
                int j = rand() % (i+1);
		ctx->permutations[i + 256] = ctx->permutations[i] = ctx->permutations[j];
                ctx->permutations[j + 256] = ctx->permutations[j] = i;
        }
}

int main(int argc, char **argv)
{
        srand(0);

        const char *symbols[] = {" ", "░", "▒", "▓", "█", "█"};
        float *pixels = malloc(sizeof(float) * 256 * 256);

        Noise2DContext n2d;
	init_noise2d(&n2d);

        for (int i = 0; i < 100; i++) {
                for (int y = 0; y < 256; y++) {
                        for (int x = 0; x < 256; x++) {
                                float v = noise2d_get(&n2d, x * 0.1f, y * 0.1f) * 0.5f + 0.5f;
                                pixels[y*256+x] = v;
                        }
                }
        }

        for (int y = 0; y < 256; y++) {
                for (int x = 0; x < 256; x++) {
			int idx = pixels[y*256+x] / 0.2f;
                        printf("%s", symbols[idx]);
		}
                printf("\n");
        }

        return 0;
}
