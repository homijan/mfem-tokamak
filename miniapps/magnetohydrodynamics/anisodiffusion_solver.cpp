//                       MFEM Example 14 - Parallel Version
//
// Compile with: make ex14p
//
// Sample runs:  mpirun -np 4 ex14p -m ../data/inline-quad.mesh -o 0
//               mpirun -np 4 ex14p -m ../data/star.mesh -o 2
//               mpirun -np 4 ex14p -m ../data/star-mixed.mesh -o 2
//               mpirun -np 4 ex14p -m ../data/star-mixed.mesh -o 2 -k 0 -e 1
//               mpirun -np 4 ex14p -m ../data/escher.mesh -s 1
//               mpirun -np 4 ex14p -m ../data/fichera.mesh -s 1 -k 1
//               mpirun -np 4 ex14p -m ../data/fichera-mixed.mesh -s 1 -k 1
//               mpirun -np 4 ex14p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex14p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex14p -m ../data/square-disc-nurbs.mesh -o 1
//               mpirun -np 4 ex14p -m ../data/disc-nurbs.mesh -rs 4 -o 2 -s 1 -k 0
//               mpirun -np 4 ex14p -m ../data/pipe-nurbs.mesh -o 1
//               mpirun -np 4 ex14p -m ../data/inline-segment.mesh -rs 5
//               mpirun -np 4 ex14p -m ../data/amr-quad.mesh -rs 3
//               mpirun -np 4 ex14p -m ../data/amr-hex.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Laplace problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include "Tschirnhaussen.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class CustomSolverMonitor : public IterativeSolverMonitor
{
public:
   CustomSolverMonitor(const ParMesh *m,
                       ParGridFunction *f) :
      pmesh(m),
      pgf(f) {}

   void MonitorSolution(int i, double norm, const Vector &x, bool final)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      int  num_procs, myid;

      MPI_Comm_size(pmesh->GetComm(),&num_procs);
      MPI_Comm_rank(pmesh->GetComm(),&myid);

      pgf->SetFromTrueDofs(x);

      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << *pgf
               << "window_title 'Iteration no " << i << "'"
               << "keys rRjlc\n" << flush;
   }

private:
   const ParMesh *pmesh;
   ParGridFunction *pgf;
};

class MatrixBxBCoefficient : public MatrixCoefficient
{
private:
  double kPara, kPerp;
  const MagneticFieldTschirnhaussen &Bfield;
public:
  MatrixBxBCoefficient(const MagneticFieldTschirnhaussen &Bfield_, 
    double kPara_ = 1e0, double kPerp_ = 1e-1) : 
  MatrixCoefficient(2), Bfield(Bfield_), kPara(kPara_), kPerp(kPerp_) {}
  void Eval(DenseMatrix &M, ElementTransformation &T, 
    const IntegrationPoint &ip)
  { 
    double x[2];
    Vector transip(x, 2);
    T.Transform(ip, transip);
    Vector B(2);
    Bfield.setB(transip, B);
    M.SetSize(2);
    // kPara * BxB + kPerp * (I - BxB) = kPerp * I + (kPara - kPerp) * BxB
    M(0, 0) = kPerp + (kPara - kPerp) * B(0) * B(0);
    M(0, 1) = (kPara - kPerp) * B(0) * B(1);
    M(1, 0) = (kPara - kPerp) * B(1) * B(0);
    M(1, 1) = kPerp + (kPara - kPerp) * B(1) * B(1);  
  }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "data/pulpo-meshes/pulpoSquare-o9-l10.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 3;
   double sigma = -1.0;
   double kappa = 1e2;//-1.0; // B-aligned mesh needs strong penalty.
   double eta = 0.0;
   double kPerp = 1e-6;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial,"
                  " -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");
   args.AddOption(&kPerp, "-kperp", "--k-perp", "Perpendicular conductivity ratio vs parallel conductivity.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code. NURBS meshes are projected to second order meshes.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ser_ref_levels' of uniform refinement. By default,
   //    or if ser_ref_levels < 0, we choose it to be the largest number that
   //    gives a final mesh with no more than 50,000 elements.
   {
      if (ser_ref_levels < 0)
      {
         ser_ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ser_ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use discontinuous finite elements of the specified order >= 0.
   FiniteElementCollection *fec = new DG_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system.
   ParLinearForm *b = new ParLinearForm(fespace); 
   ConstantCoefficient BC_cf(0.0);
   MagneticFieldTschirnhaussen Bfield(1.0);
   MatrixBxBCoefficient D_mcf(Bfield, 1e0, kPerp);
   //ConstantCoefficient D_mcf(1.0);
   FunctionCoefficient source_cf(TschirnhaussenSource_function);
   b->AddDomainIntegrator(new DomainLFIntegrator(source_cf));
   b->AddBdrFaceIntegrator(
     new DGDirichletLFIntegrator(BC_cf, D_mcf, sigma, kappa));
   b->Assemble();

   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   ParGridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After serial and parallel assembly we
   //    extract the corresponding parallel matrix A. 
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(D_mcf));
   a->AddInteriorFaceIntegrator(
     new DGDiffusionIntegrator(D_mcf, sigma, kappa));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_mcf, sigma, kappa));
   if (eta > 0)
   {
     a->AddInteriorFaceIntegrator(
       new DGDiffusionBR2Integrator(*fespace, eta));
     a->AddBdrFaceIntegrator(
       new DGDiffusionBR2Integrator(*fespace, eta));
   }
   a->Assemble();
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelProject();

   delete a;
   delete b;

   // 11. Depending on the symmetry of A, define and apply a parallel PCG or
   //     GMRES solver for AX=B using the BoomerAMG preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(*A);
   if (sigma == -1.0)
   {
      HyprePCG pcg(*A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(5000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(*amg);
      pcg.Mult(*B, *X);
   }
   else
   {
      CustomSolverMonitor monitor(pmesh, &x);
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(500);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetOperator(*A);
      gmres.SetPreconditioner(*amg);
      gmres.SetMonitor(monitor);
      gmres.Mult(*B, *X);
   }
   delete amg;

   // 12. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   x = *X;

   // 13. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 15. Free the used memory.
   delete X;
   delete B;
   delete A;

   delete fespace;
   delete fec;
   delete pmesh;

   return 0;
}
