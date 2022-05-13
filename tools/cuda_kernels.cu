


// //Define atomic_add if it's not supported
// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// __device__ double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                 __double_as_longlong(val + __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }
// #endif

//Get the CIC index from the particle position ( device function )
__device__ void Get_Indexes_CIC( double xMin, double yMin, double zMin, double dx, double dy, double dz, double pos_x, double pos_y, double pos_z, int &indx_x, int &indx_y, int &indx_z ){
  indx_x = (int) floor( ( pos_x - xMin - 0.5*dx ) / dx );
  indx_y = (int) floor( ( pos_y - yMin - 0.5*dy ) / dy );
  indx_z = (int) floor( ( pos_z - zMin - 0.5*dz ) / dz );
}

extern "C"{
//CUDA Kernel to compute the CIC density from the particles positions
__global__ void Get_Density_CIC_Kernel( int n_local, double particle_mass,  double *density_dev, 
                                        double *pos_x_dev, double *pos_y_dev, double *pos_z_dev, 
                                        double xMin, double yMin, double zMin, 
                                        double xMax, double yMax, double zMax, 
                                        double dx, double dy, double dz, 
                                        int nx, int ny, int nz, int n_ghost  ){

  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;

  int nx_g, ny_g;
  nx_g = nx + 2*n_ghost;
  ny_g = ny + 2*n_ghost;

  double pos_x, pos_y, pos_z, pMass;
  double cell_center_x, cell_center_y, cell_center_z;
  double delta_x, delta_y, delta_z;
  double dV_inv = 1./(dx*dy*dz);

  pos_x = pos_x_dev[tid];
  pos_y = pos_y_dev[tid];
  pos_z = pos_z_dev[tid];

  pMass = particle_mass * dV_inv;
  
  int indx_x, indx_y, indx_z, indx;
  Get_Indexes_CIC( xMin, yMin, zMin, dx, dy, dz, pos_x, pos_y, pos_z, indx_x, indx_y, indx_z );

  bool in_local = true;

  if ( pos_x < xMin || pos_x >= xMax ) in_local = false;
  if ( pos_y < yMin || pos_y >= yMax ) in_local = false;
  if ( pos_z < zMin || pos_z >= zMax ) in_local = false;
  if ( ! in_local  ) {
    printf(" Density CIC Error: Particle outside local domain [%f  %f  %f]  [%f %f] [%f %f] [%f %f]\n ", pos_x, pos_y, pos_z, xMin, xMax, yMin, yMax, zMin, zMax);
    return;
  }

  cell_center_x = xMin + indx_x*dx + 0.5*dx;
  cell_center_y = yMin + indx_y*dy + 0.5*dy;
  cell_center_z = zMin + indx_z*dz + 0.5*dz;
  delta_x = 1 - ( pos_x - cell_center_x ) / dx;
  delta_y = 1 - ( pos_y - cell_center_y ) / dy;
  delta_z = 1 - ( pos_z - cell_center_z ) / dz;
  indx_x += n_ghost;
  indx_y += n_ghost;
  indx_z += n_ghost;


  indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * delta_y * delta_z;
  atomicAdd( &density_dev[indx],  pMass  * delta_x * delta_y * delta_z);

  indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * (1-delta_x) * delta_y * delta_z;
  atomicAdd( &density_dev[indx], pMass  * (1-delta_x) * delta_y * delta_z);

  indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * (1-delta_y) * delta_z;
  atomicAdd( &density_dev[indx], pMass  * delta_x * (1-delta_y) * delta_z);
  //
  indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * delta_y * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass  * delta_x * delta_y * (1-delta_z) );

  indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * (1-delta_x) * (1-delta_y) * delta_z;
  atomicAdd( &density_dev[indx], pMass  * (1-delta_x) * (1-delta_y) * delta_z);

  indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass  * (1-delta_x) * delta_y * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass  * (1-delta_x) * delta_y * (1-delta_z));

  indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * (1-delta_y) * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass  * delta_x * (1-delta_y) * (1-delta_z));

  indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass * (1-delta_x) * (1-delta_y) * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass * (1-delta_x) * (1-delta_y) * (1-delta_z));

}

}

