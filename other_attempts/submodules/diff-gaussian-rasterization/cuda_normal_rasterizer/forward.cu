/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace cg = cooperative_groups;

__device__ float determinant3x3F(float matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

__device__ float determinant4x4F(float matrix[4][4]) {
    float det = 0.0f;
    float submatrix[3][3];

    for (int i = 0; i < 4; i++) {
        int subi = 0;
        for (int j = 1; j < 4; j++) {
            int subj = 0;
            for (int k = 0; k < 4; k++) {
                if (k == i) continue;
                submatrix[subi][subj] = matrix[j][k];
                subj++;
            }
            subi++;
        }
        det += (i % 2 == 0 ? 1 : -1) * matrix[0][i] * determinant3x3F(submatrix);
    }

    return det;
}

__device__ void adjugate4x4F(float matrix[4][4], float adj[4][4]) {
    float submatrix[3][3];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int subi = 0;
            for (int x = 0; x < 4; x++) {
                if (x == i) continue;
                int subj = 0;
                for (int y = 0; y < 4; y++) {
                    if (y == j) continue;
                    submatrix[subi][subj] = matrix[x][y];
                    subj++;
                }
                subi++;
            }
            adj[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * determinant3x3F(submatrix);
        }
    }
}

__device__ void inverseMatrixF(const float* h_A, float* result) {
	float matrix[4][4];
	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j ++){
			matrix[i][j] = h_A[i * 4 + j];
		}
	}
    float det = determinant4x4F(matrix);

    if (det == 0.0f) {
        // Handle non-invertible matrix
        return;
    }

    float adj[4][4];
    adjugate4x4F(matrix, adj);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i* 4 + j] = - adj[i][j] / det;
        }
    }

	for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i* 4 + j] = result[i* 4 + j] / result[15];
        }
    }
}

// Device function to compute cubic roots
__device__ void cubicRootsF(float p, float q, float r, float* roots) {
    // Coefficients for the cubic equation x^3 + ax^2 + bx + c = 0
    float a = p, b = q, c = r;
    
    // Compute Q, R, and D
    float Q = (3.0 * b - a * a) / 9.0;
    float R = (9.0 * a * b - 27.0 * c - 2.0 * a * a * a) / 54.0;
    float D = Q * Q * Q + R * R;

    if (D >= 0) { // One real root and two complex conjugate roots or a double root
        float S = cbrt(R + sqrt(D));
        float T = cbrt(R - sqrt(D));
        
        roots[0] = (S + T) - a / 3.0; // Real root
        roots[1] = -0.5 * (S + T) - a / 3.0; // Real part of complex roots
        roots[2] = roots[1]; // Real part of the other complex root
    } else { // Three real roots
        float theta = acos(R / sqrt(-Q * Q * Q));
        float sqrtQ = sqrt(-Q);
        
        roots[0] = 2.0 * sqrtQ * cos(theta / 3.0) - a / 3.0;
        roots[1] = 2.0 * sqrtQ * cos((theta + 2.0 * M_PI) / 3.0) - a / 3.0;
        roots[2] = 2.0 * sqrtQ * cos((theta + 4.0 * M_PI) / 3.0) - a / 3.0;
    }
}

// Helper function to solve a linear system using Cramer's rule
__device__ void solveLinearSystemF(float* A, float* x) {
	float B[3][3];
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			B[i][j] = A[i*3+j];
		}
	}
	float v1, v2, v3;
	v1 = 1.0;
	v2 = 0.0;
	v3 = 0.0;
    if (fabs(B[0][1] * B[1][2] - B[1][1] * B[0][2]) > 1e-10){
		v2 = (B[1][0] * B[0][2] - B[0][0] * B[1][2]) / (B[0][1] * B[1][2] - B[1][1] * B[0][2]);
	}
	else if (fabs(B[0][1] * B[2][2] - B[2][1] * B[0][2]) > 1e-10){
		v2 = (B[2][0] * B[0][2] - B[0][0] * B[2][2]) / (B[0][1] * B[2][2] - B[2][1] * B[0][2]);
	}
	else if (fabs(B[1][1] * B[2][2] - B[2][1] * B[1][2]) > 1e-10){
		v2 = (B[2][0] * B[1][2] - B[1][0] * B[2][2]) / (B[1][1] * B[2][2] - B[2][1] * B[1][2]);
	}
	else{
		v2 = 0.0;
	}

	for (int i = 0; i < 3; i++){
		if (B[i][2] > 1e-10){
			v3 = (-B[i][0] - B[i][1] * v2) / B[i][2];
			break;
		}
	}

	x[0] = v1;
	x[1] = v2;
	x[2] = v3;
}

// Kernel to compute eigenvalues and eigenvectors
__device__ void computeEigenValuesVectorsF(const float* matrix, float* eigenvalues, float* eigenvectors) {
    // Load the matrix elements from global memory
    float a = matrix[0], b = matrix[1], c = matrix[2];
    float d = matrix[3], e = matrix[4], f = matrix[5];
    float g = matrix[6], h = matrix[7], i = matrix[8];

    // Compute characteristic polynomial coefficients
    float p = -(a + e + i);
    float q = a * e + a * i + e * i - b * f - c * h - d * g;
    float r = a * e * i + b * f * g + c * d * h - a * f * h - b * d * i - c * e * g;

    // Array to store eigenvalues
    float roots[3];

    // Compute eigenvalues
    cubicRootsF(p, q, r, roots);

    // Store eigenvalues back to global memory
    eigenvalues[0] = roots[0];
    eigenvalues[1] = roots[1];
    eigenvalues[2] = roots[2];

        // Now compute eigenvectors for each eigenvalue
    for (int j = 0; j < 3; ++j) {
        float lambda = roots[j];
        
        // Create the matrix (A - lambda * I)
        float A_lambdaI[9] = {
            a - lambda, b, c,
            d, e - lambda, f,
            g, h, i - lambda
        };

        // Resultant eigenvector for eigenvalue lambda
        float v[3] = { 0.0, 0.0, 0.0 };

        // Solve the linear system to find the eigenvector
        solveLinearSystemF(A_lambdaI, v);

        // normalizeF the eigenvector (to prevent numerical instability)
        float norm = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (norm > 1e-10) {
            v[0] /= norm;
            v[1] /= norm;
            v[2] /= norm;
        }

		// printf("Eigen Vector found: %f, %f, %f\n", v[0], v[1], v[2]);

        // Store eigenvector back to global memory
        eigenvectors[j * 3 + 0] = v[0];
        eigenvectors[j * 3 + 1] = v[1];
        eigenvectors[j * 3 + 2] = v[2];
    }
}

__device__ float dotProductF(glm::vec3 mat1, glm::vec3 mat2){
	return (float)(mat1.x * mat2.x + mat1.y * mat2.y + mat1.z * mat2.z);
}

__device__ glm::vec3 normalizeF(glm::vec3 vec){
	float norm = sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
	if (norm > 1e-10){
		vec /= norm;
	}
	return vec;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSHF(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}


// Helper function to compute the inverse of a 3x3 matrix
__device__ glm::mat3 inverse3x3_temp(const float* m) {
    glm::mat3 mat = glm::mat3(m[0], m[1], m[2],
                              m[3], m[4], m[5],
                              m[6], m[7], m[8]);
    return glm::inverse(mat);
}

// Forward method for converting the input spacial param
// of each Gaussian to a simple RGB color using its normal vector.
__device__ void computeColorFromNormalF(glm::vec3* color, glm::vec3 mean,const float* cov3D, const float* viewmatrix, const float* projmatrix, float2 pixf, int screenWidth, int screenHeight)
{
    int m = 3;
    float w[3];  // Array to hold eigenvalues

    float h_V[9];
	computeEigenValuesVectorsF(cov3D, w, h_V);

	// eigenvalues: w (3)
	// eigenvectors: h_V (3,3)
    glm::vec3 eigenVectors[3];
	eigenVectors[0].x = h_V[0];
	eigenVectors[0].y = h_V[1];
	eigenVectors[0].z = h_V[2];
	eigenVectors[1].x = h_V[3];
	eigenVectors[1].y = h_V[4];
	eigenVectors[1].z = h_V[5];
	eigenVectors[2].x = h_V[6];
	eigenVectors[2].y = h_V[7];
	eigenVectors[2].z = h_V[8];

	int maxIdx = 0;
	float maxEigenValue = w[0];
	if (maxEigenValue < w[1]){
		maxIdx = 1;
		maxEigenValue = w[1];
	}
	if (maxEigenValue < w[2]){
		maxIdx = 2;
		maxEigenValue = w[2];
	}

	if (maxIdx > 0){
		float temp = w[0];
		glm::vec3 tempVec = eigenVectors[0];
		w[0] = w[maxIdx];
		eigenVectors[0] = eigenVectors[maxIdx];
		w[maxIdx] = temp;
		eigenVectors[maxIdx] = tempVec;
	}

	float radius = max(w[1], w[2]);
	glm::vec3 primAxis = normalizeF(eigenVectors[maxIdx]);

	glm::vec3 minAxis = eigenVectors[1];
	if (minAxis.x * minAxis.x + minAxis.y * minAxis.y + minAxis.z * minAxis.z > eigenVectors[2].x * eigenVectors[2].x + eigenVectors[2].y * eigenVectors[2].y + eigenVectors[2].z * eigenVectors[2].z){
		minAxis = eigenVectors[2];
	}

	// if (pixf.x < 5 && pixf.y < 5){
	// 	printf("Color: %f, %f, %f\n", result.r, result.g, result.b);
	// }

	// if (pixf.x == 1 && pixf.y == 1){
	// 	printf("VIew Matrix 1: %f, %f, %f, %f\n",viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3]);
	// 	printf("VIew Matrix 2: %f, %f, %f, %f\n",viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7]);
	// 	printf("VIew Matrix 3: %f, %f, %f, %f\n",viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11]);
	// 	printf("VIew Matrix 4: %f, %f, %f, %f\n",viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]);
	// }
	float vm[16];
	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j++){
			vm[i*4+j] = viewmatrix[j*4+i];
		}
	}

	glm::vec3 cameraPos;
	float vmInverse[16];
	inverseMatrixF(vm, vmInverse);

	cameraPos = glm::vec3(vmInverse[3], vmInverse[7], vmInverse[11]);

	if (pixf.x == 1 && pixf.y == 1){
		printf("camera position: %f, %f, %f\n",cameraPos.x, cameraPos.y, cameraPos.z);
	}

	float x_pixel = pixf.x;
	float y_pixel = pixf.y;
	float x_ndc = (2.0 * x_pixel) / float(screenWidth) - 1.0;
    float y_ndc = 1.0 - (2.0 * y_pixel) / float(screenHeight);

    // Clip space coordinates
	float rayClipFar[4] = {x_ndc, y_ndc, 0.0, 1.0};

	float pm[16];
	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j++){
			pm[i*4+j] = projmatrix[j*4+i];
		}
	}

    // Transform to camera space
	float pmInverse[16];
	inverseMatrixF(pm, pmInverse);

	// if (pixf.x == 1 && pixf.y == 1){
	// 	printf("Proj Matrix 1: %f, %f, %f, %f\n",projmatrix[0], projmatrix[1], projmatrix[2], projmatrix[3]);
	// 	printf("Proj Matrix 2: %f, %f, %f, %f\n",projmatrix[4], projmatrix[5], projmatrix[6], projmatrix[7]);
	// 	printf("Proj Matrix 3: %f, %f, %f, %f\n",projmatrix[8], projmatrix[9], projmatrix[10], projmatrix[11]);
	// 	printf("Proj Matrix 4: %f, %f, %f, %f\n",projmatrix[12], projmatrix[13], projmatrix[14], projmatrix[15]);
	// }

    float rayCameraFar[4];
	for (int i = 0; i < 4; i ++){
		float temp_far = 0.0;
		for (int j = 0; j < 4; j ++){
			temp_far = temp_far + pmInverse[i * 4 + j] * rayClipFar[j];
		}
		rayCameraFar[i] = temp_far;
	}

	for (int i = 0; i < 4; i++){
		rayCameraFar[i] = rayCameraFar[i] / rayCameraFar[3];
	}

    rayCameraFar[3] = 1.0;

    // Transform to world space
    float rayWorldFarRaw[4];
	for (int i = 0; i < 4; i ++){
		float temp_far = 0.0;
		for (int j = 0; j < 4; j ++){
			temp_far = temp_far + vmInverse[i * 4 + j] * rayCameraFar[j];
		}
		rayWorldFarRaw[i] = temp_far;
	}

	for (int i = 0; i < 4; i++){
		rayWorldFarRaw[i] = rayWorldFarRaw[i] / rayWorldFarRaw[3];
	}

	// if (pixf.x == 25 && pixf.y == 25){
	// 	printf("Ray World Far: %f, %f, %f\n",rayWorldFarRaw[0], rayWorldFarRaw[1], rayWorldFarRaw[2]);
	// }

	glm::vec3 rayWorld = {rayWorldFarRaw[0] - cameraPos.x, rayWorldFarRaw[1] - cameraPos.y, rayWorldFarRaw[2] - cameraPos.z};
	rayWorld = normalizeF(rayWorld);

	glm::vec3 resultA = (rayWorld + 1.f) * 0.5f;
	color->x = max(resultA.r, 0.0f);
	color->y = max(resultA.g, 0.0f);
	color->z = max(resultA.b, 0.0f);

	return;

	// if (pixf.x < 5 && pixf.y < 5){
	// 	printf("ray direction: %f, %f, %f\n",rayWorld.x, rayWorld.y, rayWorld.z);
	// }

    glm::vec3 E = cameraPos - mean;

    // glm::vec3 V = rayWorld - primAxis * dotProductF(rayWorld, primAxis);
    // glm::vec3 W = OC - primAxis * dotProductF(OC, primAxis);

    float a = dotProductF(rayWorld, rayWorld) - dotProductF(rayWorld,primAxis) * dotProductF(rayWorld,primAxis);
    float b = 2.0f * (dotProductF(rayWorld,E) - dotProductF(rayWorld,primAxis)*dotProductF(E, primAxis));
    float c = dotProductF(E,E) - dotProductF(E,primAxis) * dotProductF(E,primAxis) - radius * radius;

    float discriminant = b * b - 4.0f * a * c;
	if (discriminant < 1e-10){
		color->x = 0.0;
		color->y = 0.0;
		color->z = 0.0;
		return;
	}
    float t = (-b - sqrtf(discriminant)) / (2.0f * a); // Choose the smaller positive root
	if (t < 1e-10){
		t = (-b + sqrtf(discriminant)) / (2.0f * a);
	}

	// if (pixf.x < 5 && pixf.y < 5){
	// 	printf("Calculated t: %f\n",t);
	// }

    glm::vec3 P = cameraPos + rayWorld * t;
    glm::vec3 Q = mean + primAxis * dotProductF(P - mean, primAxis);
    glm::vec3 N = normalizeF(P - Q);

	glm::vec3 result = (N + 1.f) * 0.5f;

	// if (pixf.x < 5 && pixf.y < 5){
	// 	printf("Color: %f, %f, %f\n", result.r, result.g, result.b);
	// }

	color->x = max(result.r, 0.0f);
	color->y = max(result.g, 0.0f);
	color->z = max(result.b, 0.0f);
}


// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2DF(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3DF(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// normalizeF quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float3* points_xyz,
	float* depths,
	std::pair<float3,float3>* cov6,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3DF(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2DF(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise calculate
	// color form normal vector
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSHF(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	points_xyz[idx] = p_orig;

	cov6[idx].first.x = cov3D[0];
	cov6[idx].first.y = cov3D[1];
	cov6[idx].first.z = cov3D[2];
	cov6[idx].second.x = cov3D[3];
	cov6[idx].second.y = cov3D[4];
	cov6[idx].second.z = cov3D[5];

	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ points_xyz,
	const std::pair<float3,float3>* __restrict__ cov6,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* viewmatrix,
	const float* projmatrix)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ std::pair<float3,float3> collected_cov6[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;

			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_xyz[block.thread_rank()] = points_xyz[coll_id];
			collected_cov6[block.thread_rank()].first = cov6[coll_id].first;
			collected_cov6[block.thread_rank()].second = cov6[coll_id].second;

			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			glm::vec3 mean;
			mean.x = collected_xyz[j].x;
			mean.y = collected_xyz[j].y;
			mean.z = collected_xyz[j].z;

			float cov3D[9];
			cov3D[0] = collected_cov6[j].first.x;
			cov3D[1] = collected_cov6[j].first.y;
			cov3D[2] = collected_cov6[j].first.z;
			cov3D[3] = collected_cov6[j].first.y;
			cov3D[4] = collected_cov6[j].second.x;
			cov3D[5] = collected_cov6[j].second.y;
			cov3D[6] = collected_cov6[j].first.z;
			cov3D[7] = collected_cov6[j].second.y;
			cov3D[8] = collected_cov6[j].second.z;

			glm::vec3 color;
			computeColorFromNormalF(&color, mean, cov3D, viewmatrix, projmatrix, pixf, W, H);

			color.x = 0.5;

			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += color[ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD_NORMAL::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float3* points_xyz,
	const std::pair<float3,float3>* cov6,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* viewmatrix,
	const float* projmatrix)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		points_xyz,
		cov6,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		viewmatrix,
		projmatrix);
}

void FORWARD_NORMAL::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float3* means3Ds,
	float* depths,
	std::pair<float3,float3>* cov6,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		means3Ds,
		depths,
		cov6,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}