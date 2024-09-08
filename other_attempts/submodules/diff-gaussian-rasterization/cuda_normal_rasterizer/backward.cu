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

#include "backward.h"
#include "auxiliary.h"
#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cg = cooperative_groups;

// Device function to compute cubic roots
__device__ void cubicRootsB(float p, float q, float r, float* roots) {
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
__device__ void solveLinearSystemB(float* A, float* B, float* x) {
    // Calculate the determinant of A
    float detA = A[0] * (A[4] * A[8] - A[5] * A[7]) -
                  A[1] * (A[3] * A[8] - A[5] * A[6]) +
                  A[2] * (A[3] * A[7] - A[4] * A[6]);

    // If determinant is zero, we have either no solution or infinite solutions.
    if (fabs(detA) < 1e-10) {
        x[0] = x[1] = x[2] = 0.0; // Set to zero vector for now
        return;
    }

    // Compute determinants of matrices where we replace each column with B
    float detA1 = B[0] * (A[4] * A[8] - A[5] * A[7]) -
                   A[1] * (B[1] * A[8] - B[2] * A[7]) +
                   A[2] * (B[1] * A[7] - B[2] * A[4]);

    float detA2 = A[0] * (B[1] * A[8] - B[2] * A[7]) -
                   B[0] * (A[3] * A[8] - A[5] * A[6]) +
                   A[2] * (A[3] * B[2] - A[6] * B[1]);

    float detA3 = A[0] * (A[4] * B[2] - A[5] * B[1]) -
                   A[1] * (A[3] * B[2] - A[6] * B[1]) +
                   B[0] * (A[3] * A[7] - A[6] * A[4]);

    // Calculate the solutions (Cramer's rule)
    x[0] = detA1 / detA;
    x[1] = detA2 / detA;
    x[2] = detA3 / detA;
}

// Kernel to compute eigenvalues and eigenvectors
__device__ void computeEigenValuesVectorsB(const float* matrix, float* eigenvalues, float* eigenvectors) {
    // Example for a single 3x3 matrix
    int idx = threadIdx.x;

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
    cubicRootsB(p, q, r, roots);

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

        // We need a vector B = [0, 0, 0] since we are solving (A - lambda*I)v = 0
        float B[3] = { 0.0, 0.0, 0.0 };

        // Resultant eigenvector for eigenvalue lambda
        float v[3] = { 0.0, 0.0, 0.0 };

        // Solve the linear system to find the eigenvector
        solveLinearSystemB(A_lambdaI, B, v);

        // normalizeB the eigenvector (to prevent numerical instability)
        float norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (norm > 1e-10) {
            v[0] /= norm;
            v[1] /= norm;
            v[2] /= norm;
        }

        // Store eigenvector back to global memory
        eigenvectors[j * 3 + 0] = v[0];
        eigenvectors[j * 3 + 1] = v[1];
        eigenvectors[j * 3 + 2] = v[2];
    }
}

__device__ float dotProductB(glm::vec3 mat1, glm::vec3 mat2){
	return (float)(mat1.x * mat2.x + mat1.y * mat2.y + mat1.z * mat2.z);
}

__device__ glm::vec3 normalizeB(glm::vec3 vec){
	float norm = sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
	vec /= norm;
	return vec;
}

__device__ float determinant3x3B(float matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

__device__ float determinant4x4B(float matrix[4][4]) {
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
        det += (i % 2 == 0 ? 1 : -1) * matrix[0][i] * determinant3x3B(submatrix);
    }

    return det;
}

__device__ void adjugate4x4B(float matrix[4][4], float adj[4][4]) {
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
            adj[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * determinant3x3B(submatrix);
        }
    }
}

__device__ void inverseMatrixB(const float* h_A, float* result) {
	float matrix[4][4];
	for (int i = 0; i < 4; i ++){
		for (int j = 0; j < 4; j ++){
			matrix[i][j] = h_A[i * 4 + j];
		}
	}
    float det = determinant4x4B(matrix);

    if (det == 0.0f) {
        // Handle non-invertible matrix
        return;
    }

    float adj[4][4];
    adjugate4x4B(matrix, adj);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i* 4 + j] = adj[i][j] / det;
        }
    }
}


// Forward method for converting the input spacial param
// of each Gaussian to a simple RGB color using its normal vector.
__device__ void computePrimAxisAndRB(float4* PandR, const float* cov3D)
{
    int m = 3;
    float w[3];  // Array to hold eigenvalues

    float h_V[9];
	computeEigenValuesVectorsB(cov3D, w, h_V);

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
		w[0] = w[maxIdx];
		w[maxIdx] = temp;
	}

	float radius = max(w[1], w[2]);
	glm::vec3 primAxis = eigenVectors[maxIdx];

	PandR->x = primAxis.x;
	PandR->y = primAxis.y;
	PandR->z = primAxis.z;
	PandR->w = radius;
}


// Forward method for converting the input spacial param
// of each Gaussian to a simple RGB color using its normal vector.
__device__ glm::vec3 computeSurfacePointB(glm::vec3 mean, glm::vec3 primAxis, float radius, const float* viewmatrix, const float* projmatrix, float2 pixf, int screenWidth, int screenHeight)
{
	float vmInverse[16];
	float origin[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	float cameraPosition[4];
	inverseMatrixB(viewmatrix, vmInverse);

	for (int i = 0; i < 4; i ++){
		float temp = 0.0;
		for (int j = 0; j < 4; j ++){
			temp = temp + vmInverse[i * 4 + j] * origin[j];
		}
		cameraPosition[i] = temp;
	}
	glm::vec3 cameraPos;
	cameraPos.x = cameraPosition[0];
	cameraPos.y = cameraPosition[1];
	cameraPos.z = cameraPosition[2];

	float x_pixel = pixf.x;
	float y_pixel = pixf.y;
	float x_ndc = (2.0 * float(x_pixel)) / float(screenWidth) - 1.0;
    float y_ndc = 1.0 - (2.0 * float(y_pixel)) / float(screenHeight);

    // Clip space coordinates
    float rayClip[4] = {x_ndc, y_ndc, -1.0, 1.0};

    // Transform to camera space
	float pmInverse[16];
	inverseMatrixB(projmatrix, pmInverse);
    float rayCamera[4];
	for (int i = 0; i < 4; i ++){
		float temp = 0.0;
		for (int j = 0; j < 4; j ++){
			temp = temp + pmInverse[i * 4 + j] * rayClip[j];
		}
		rayCamera[i] = temp;
	}

	for (int i = 0; i < 3; i++){
		rayCamera[i] = rayCamera[i] / rayCamera[3];
	}
    rayCamera[3] = 0.0; // Perspective divide

    // Transform to world space
    float rayWorldRaw[3];
	// vmInverse * rayCamera

	for (int i = 0; i < 3; i ++){
		float temp = 0.0;
		for (int j = 0; j < 4; j ++){
			temp = temp + vmInverse[i * 4 + j] * rayCamera[j];
		}
		rayWorldRaw[i] = temp;
	}
	glm::vec3 rayWorld;
	float norm = rayWorldRaw[0] * rayWorldRaw[0] + rayWorldRaw[1] * rayWorldRaw[1] + rayWorldRaw[2] * rayWorldRaw[2];
	norm = sqrtf(norm);
	rayWorld.x = rayWorldRaw[0] / norm;
	rayWorld.y = rayWorldRaw[1] / norm;
	rayWorld.z = rayWorldRaw[2] / norm;

	// float3 cameraPos;
	// float3 rayWorld;
	// const glm::vec3* mean;

	// cameraPos + t * rayWorld = mean + 
    glm::vec3 OC = cameraPos - mean;

    glm::vec3 V = rayWorld - primAxis * dotProductB(rayWorld, primAxis);
    glm::vec3 W = OC - primAxis * dotProductB(OC, primAxis);

    float a = dotProductB(V, V);
    float b = 2.0f * dotProductB(V, W);
    float c = dotProductB(W, W) - radius * radius;

    float discriminant = b * b - 4.0f * a * c;
    float t = (-b - sqrtf(discriminant)) / (2.0f * a); // Choose the smaller positive root

    glm::vec3 P = cameraPos + rayWorld * t;
	return P;
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSHB(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDAB(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] += (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] += (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] += (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] += 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] += 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] += 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] += 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx].x += dL_dmean.x;
	dL_dmeans[idx].y += dL_dmean.y;
	dL_dmeans[idx].z += dL_dmean.z;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3DB(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalizeBd quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalizeBd quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSHB(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3DB(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ points_xyz,
	const std::pair<float3,float3>* __restrict__ cov6,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	glm::vec3* __restrict__ dL_dmeans3D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dcov3D,
	const float* viewmatrix,
	const float* projmatrix)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ std::pair<float3,float3> collected_cov6[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalizeBd 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			
			collected_xyz[block.thread_rank()] = points_xyz[coll_id];
			collected_cov6[block.thread_rank()].first = cov6[coll_id].first;
			collected_cov6[block.thread_rank()].second = cov6[coll_id].second;

			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;


			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			float dL_dnormal[C];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

				// Using normal vector as output color, record dL_dcolors as dL_dNormal		
				// atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				dL_dnormal[ch] = dchannel_dcolor * dL_dchannel;
			}

			float cov3d[6];
			cov3d[0] = collected_cov6[j].first.x;
			cov3d[1] = collected_cov6[j].first.y;
			cov3d[2] = collected_cov6[j].first.z;
			cov3d[3] = collected_cov6[j].second.x;
			cov3d[4] = collected_cov6[j].second.y;
			cov3d[5] = collected_cov6[j].second.z;

			float4 primAxisAndR;
			computePrimAxisAndRB(&primAxisAndR, cov3d);

			glm::vec3 primAxis = glm::vec3(primAxisAndR.x, primAxisAndR.y, primAxisAndR.z);
			float radius = primAxisAndR.w;

			glm::vec3 dL_dN;
			dL_dN.x = dL_dnormal[0];
			dL_dN.y = dL_dnormal[1];
			dL_dN.z = dL_dnormal[2];
			float dot_dLdN_a = dotProductB(dL_dN, primAxis);
			float3 dL_dC_local = {-dot_dLdN_a * primAxis.x, -dot_dLdN_a * primAxis.y, -dot_dLdN_a * primAxis.z};
			// Update gradient regarding to 3D means
			atomicAdd(&dL_dmeans3D[global_id].x, dL_dC_local.x);
			atomicAdd(&dL_dmeans3D[global_id].y, dL_dC_local.y);
			atomicAdd(&dL_dmeans3D[global_id].z, dL_dC_local.z);

			glm::vec3 means3D = glm::vec3(collected_xyz[j].x, collected_xyz[j].y, collected_xyz[j].z);
			glm::vec3 p = computeSurfacePointB(means3D, primAxis, radius, viewmatrix, projmatrix, pixf, W, H);
			glm::vec3 c = glm::vec3(collected_xyz[j].x, collected_xyz[j].y, collected_xyz[j].z);
			glm::vec3 pc = p - c;

			// Derivative of the normal with respect to covariance eigenvector
			glm::vec3 dN_da = normalizeB(glm::vec3(
				-2 * primAxis.x * dotProductB(pc, primAxis) + 2 * pc.x,
				-2 * primAxis.y * dotProductB(pc, primAxis) + 2 * pc.y,
				-2 * primAxis.z * dotProductB(pc, primAxis) + 2 * pc.z));
			float dot_dLdN_dNda = dotProductB(dL_dN, dN_da);

			// Update dL / dCov3D
			atomicAdd(&(dL_dcov3D[global_id * 6 + 0]), dot_dLdN_dNda * primAxis.x * primAxis.x);
			atomicAdd(&(dL_dcov3D[global_id * 6 + 1]), dot_dLdN_dNda * primAxis.x * primAxis.y);
			atomicAdd(&(dL_dcov3D[global_id * 6 + 2]), dot_dLdN_dNda * primAxis.x * primAxis.z);
			atomicAdd(&(dL_dcov3D[global_id * 6 + 3]), dot_dLdN_dNda * primAxis.y * primAxis.x);
			atomicAdd(&(dL_dcov3D[global_id * 6 + 4]), dot_dLdN_dNda * primAxis.y * primAxis.y);
			atomicAdd(&(dL_dcov3D[global_id * 6 + 5]), dot_dLdN_dNda * primAxis.z * primAxis.z);

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD_NORMAL::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDAB << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD_NORMAL::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float3* means3D,
	const std::pair<float3,float3>* cov33,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	glm::vec3* dL_dmeans3D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dcov3D,
	const float* viewmatrix,
	const float* projmatrix)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		means3D,
		cov33,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dmeans3D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dcov3D,
		viewmatrix,
		projmatrix
		);
}