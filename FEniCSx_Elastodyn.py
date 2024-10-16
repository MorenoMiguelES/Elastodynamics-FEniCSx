"""
ELASTODYNAMICS: Generalised-alpha method, traction forces / displacement loading.
Author: Dr. Miguel Angel Moreno-Mateos
"""
#Import: not necessary to specify it later
from __future__ import print_function 
import numpy as np
from numpy import array
import  dolfinx
import ufl
from mpi4py import MPI
import dolfinx.io
import math
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import petsc4py
import matplotlib.pyplot as plt         # Module matplotlib for plotting
#from dolfinx import default_scalar_type
from petsc4py import PETSc
default_scalar_type = PETSc.ScalarType
from dolfinx import fem

name = "Output.xdmf" # Name of the output file
name1 = "Output.h5"

j,k,l,m,n,o,p,q = ufl.indices(8)

T = 40  # Final time
T_ramp=0.3 # SPLIT: 1) Application Dirichlet BC / traction forces, 2) Free oscillation
steps = 0 #Inizialization of the variable.

# Time parameters
tiempo = 0.0
tot_steps = 1000;
dt = T / tot_steps

#Import mesh & subdomains from .msh file:
filename="Box"
from dolfinx.io import gmshio
mesh, cell_markers, facet_markers = gmshio.read_from_msh(filename+'.msh', MPI.COMM_WORLD, gdim=3)

metadata = {"quadrature_degree": 2}
dx = ufl.Measure('dx')(domain=mesh, subdomain_data=cell_markers, metadata=metadata) #Integration measures for each subdomain

#Function spaces to interpolate the results and plot in Paraview.
TT = dolfinx.fem.TensorFunctionSpace(mesh,('DG',1)) #To interpolate 2-order tensors
VV = dolfinx.fem.VectorFunctionSpace(mesh,('DG',1)) #To eventually interpolate 1-order tensors
SS = dolfinx.fem.FunctionSpace(mesh,('DG',1)) #To interpolate scalar fields

# FUNCTION SPACES
P1 = ufl.VectorElement("CG",mesh.ufl_cell(),2) #Displacements
V = dolfinx.fem.FunctionSpace(mesh, P1)

# FUNCTIONS IN THE FUNCTION SPACE V (Function - Trial - Test).
u = dolfinx.fem.Function(V) #Unkown.
u_old = dolfinx.fem.Function(V)
v_old = dolfinx.fem.Function(V)
a_old = dolfinx.fem.Function(V)
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dd = u.geometric_dimension()  # space dimension

########## BOUNDARY CONDITIONS ############################################
###### Encastre Displacement:
def bnd_encastre(x):
    tol=1e-4
    return np.isclose(x[2],0)
bnd_encastre_dofs0 = dolfinx.fem.locate_dofs_geometrical(V,bnd_encastre)
u_encastre_vect0 = np.array((0,) * mesh.geometry.dim, dtype=default_scalar_type)
bc_boundar = dolfinx.fem.dirichletbc(u_encastre_vect0,bnd_encastre_dofs0,V)

###### Option A: Displacement load: #######################################
def bnd_point(x):
    return np.isclose(x[2],1) & np.isclose(x[0],0.04)
dispmax = 0.2
class MyDisp:
    def __init__(self):
        self.t = 0.0
    def eval(self,x):
        #Add some spatial/temporal variation here:
        #x.shape[1]: number of columns.
        if True:
            return np.full(x.shape[1], dispmax * (self.t)/(T_ramp))
        else:
            return np.full(x.shape[1], dispmax * (self.t)/(T_ramp))
V_real = dolfinx.fem.FunctionSpace(mesh, ("CG",2))
loaddisp_ = MyDisp()   
loaddisp_.t = 0.0
loaddisp = dolfinx.fem.Function(V_real)
loaddisp.interpolate(loaddisp_.eval)
bnd_load_dofs0 = dolfinx.fem.locate_dofs_geometrical((V.sub(0),V.sub(0).collapse()[0]),bnd_point)
bc_dispload = dolfinx.fem.dirichletbc(loaddisp,bnd_load_dofs0,V.sub(0))
#########################################################################

# Definition of 2 problems: bcu1 with Dirichlet BC on right side,
#bcu2 without them for free oscillation. If traction forces, it is not necessary.
bcu1 = [bc_boundar]#, bc_dispload] #Step 1: Dirichlet BC
bcu2 = [bc_boundar] #Step 2: Oscillation

########## Option B:TRACTION FORCES #####################################
def bnd_side(x):
    return np.isclose(x[2],1)
forcemax = 1
t_ = dolfinx.fem.Constant(mesh, 0.0)
traction = ufl.as_vector([forcemax * (t_)/(T_ramp),0,0])
right_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, bnd_side)
facet_tag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, right_facets, 1)
dss = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tag, subdomain_id=1)#, metadata=metadata)
#######################################################################

########### GENERALISED-ALPHA METHOD (time discretisation) ############
# alpha_m = 0 # in [0,1]
# alpha_f = 0 # in [0,1]
# gamma = 0.5 + alpha_m - alpha_f
# beta = (gamma + 0.5)**2 / 4
#######################################################################
    
########### NEWMARK METHOD (time discretisation) ######################
# alpha_m = alpha_f = 0: recovers Newmark method.
alpha_m = 0 # in [0,1]
alpha_f = 0 # in [0,1]
# HIGH-FREQUENCY DISSIPATION (NUMERICAL DAMPING) IN NEWMARK METHOD:
xi = 0.1 # in [0,0.5]
gamma = 1#0.5 + xi
beta = 0.5#0.25*(1+xi)**2
# for xi a parameter that introduces numerical damping of high frequencies. If xi = 0, no damping.
#######################################################################

########### RAYLEIGH DAMPING ##########################################
# Note: Not necessary for high-frequency dissipation if gamma and beta are properly chosen.
eta_m = 0 #1e-4
eta_k = 0 #1e-4
#######################################################################

def avg(x_old,x_new,alpha):
    return alpha*x_old + (1-alpha)*x_new
    
def update_a(u,u_old,v_old,a_old):
    dt_ = dt
    beta_ = beta
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old
    
def update_v(a,u_old,v_old,a_old):
    dt_ = dt
    gamma_ = gamma
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)
    
a_new = update_a(u,u_old,v_old,a_old)
v_new = update_v(a_new,u_old,v_old,a_old)
######################################################################

## PRIMARY FIELDS
dd = len(u) # Spatial dimension
I = ufl.Identity(dd) # Identity tensor
F = I + ufl.grad(avg(u_old,u,alpha_f)) # Deformation gradient from current time step
Fv = I + ufl.grad(avg(v_old,v_new,alpha_f)) # Deformation gradient from current time step
CG = ufl.dot(F.T,F) # Right Cauchy-Green (CG) tensor
BG = ufl.dot(F,F.T) # Left Cauchy-Green (CG) tensor

## CONSTITUTIVE MODEL:
######## YEOH-MODEL (COMPRESSIBLE) ###################################################
C10 = [0.5*1000, 0]
C20 = [0, 0]
C30 = [0, 0]
poisson = [0.3, 0]
kappa = [2*(2*C10[0])*(1+poisson[0])/3/(1-2*poisson[0]), 2*(2*C10[0])*(1+poisson[0])/3/(1-2*poisson[0])]
#density:
rho = 1.0
def Piso(F,i):
    j,k,l,m = ufl.indices(4)
    Id = ufl.Identity(3)
    FinvT = ufl.inv(F).T
    J = ufl.det(F)
    A1 = ufl.as_tensor(Id[j,l]*Id[k,m],(j,k,l,m))
    A2 = ufl.as_tensor(-1/3*FinvT[j,k]*F[l,m],(j,k,l,m))   
    Pfourth = A1+A2
    Fiso = 1/ufl.det(F)**(1/3)*F
    I1 = ufl.tr(ufl.dot(Fiso.T,Fiso) ) 
    P_yeoh=  2* ( C10[i] + C20[i] *2* (I1-3)**1 + C30[i] *3 * (I1-3)**2 )* Fiso
    P = 1/(J**(1/3))*ufl.as_tensor(Pfourth[j,k,l,m]*P_yeoh[l,m],(j,k))
    return P
def Pvol(F,i): #volumetric part
    J = ufl.det(F)
    Pvol= kappa[i]*J*(J-1)*ufl.inv(F).T
    return Pvol
########################################################################################

######### NEO-HOOKEAN ##################################################################
#def Piso(F,i):
#    j,k,l,m = indices(4)
#    Id = Identity(3)
#    FinvT = inv(F).T
#    A1 = as_tensor(Id[j,l]*Id[k,m],(j,k,l,m))
#    A2 = as_tensor(-1/3*FinvT[j,k]*F[l,m],(j,k,l,m))
#    Pfourth = A1+A2
#    Fiso = 1/det(F)**(1/3)*F
#    Piso_ = G[i]*Fiso*(1+b_[i]/n_[i]*(tr(dot(Fiso.T,Fiso))-3))**(n_[i]-1)
#    return 1/(det(F)**(1/3))*as_tensor(Pfourth[j,k,l,m]*Piso_[l,m],(j,k))
#def Pvol(F,i):
#    return kappa[i]*det(F)*(det(F)-1)*inv(F).T
#G = [9e-3,0]
#n_ = [1.0, 0]
#b_ = [1.0, 0]
#poisson=[0.495,0,0]
#kappa = [2*G[0]*(1+poisson[0])/3/(1-2*poisson[0]),0,0]
########################################################################################

####### VARIATIONAL PROBLEM - WEAK FORM ################################################
Res_u = (ufl.inner(Piso(F, 0), ufl.grad(v))\
    + ufl.inner(Pvol(F, 0), ufl.grad(v))\
    + eta_m*rho*ufl.inner(avg(v_old,v_new,alpha_m),v) + eta_k*(ufl.inner(Piso(Fv, 0), ufl.grad(v)) + ufl.inner(Pvol(Fv, 0), ufl.grad(v)))
    + rho*ufl.inner(avg(a_old,a_new,alpha_m),v)\
    ) * dx\
    - ufl.inner(traction,v) * dss(1)

# Setup Non-linear variational problem
problem_ugi1 = fem.petsc.NonlinearProblem(Res_u,u,bcu1)
from dolfinx import nls
solver_problem_ugi1 = nls.petsc.NewtonSolver(mesh.comm, problem_ugi1)
solver_problem_ugi1.atol = 1e-8
solver_problem_ugi1.rtol = 1e-8
solver_problem_ugi1.convergence_criterion = "incremental"

problem_ugi2 = fem.petsc.NonlinearProblem(Res_u,u,bcu2)
solver_problem_ugi2 = nls.petsc.NewtonSolver(mesh.comm, problem_ugi2)
solver_problem_ugi2.atol = 1e-8
solver_problem_ugi2.rtol = 1e-8
solver_problem_ugi2.convergence_criterion = "incremental"

import os
if os.path.exists(name):
    os.remove(name)
    os.remove(name1)

# Save results into an .xdmf
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
file_results = dolfinx.io.XDMFFile(MPI.COMM_WORLD,name,'w')

# Write the mesh and subdomains info
file_results.write_mesh(mesh)
cell_markers.name = "Phases"
file_results.write_meshtags(cell_markers)

##### EVALUATE DISPLACEMENT AT A POINT ########################
time_vec = []
disp_x_point1 = []
point1 = np.array([[0.0,0.0,1]])
disp1 = []
tree = dolfinx.geometry.BoundingBoxTree(mesh, 3)
cells = []
points_on_proc = []
cell_candidates = dolfinx.geometry.compute_collisions(tree, point1) # Find cells whose bounding-box collide with the the points
colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point1) # Choose one of the cells that contains the point
for i, point in enumerate(point1):
    if len(colliding_cells.links(i))>0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
print(points_on_proc)
#################################################################

###### TIME-INCREMENTAL SOLVING LOOP ########################
# Solve for each value using the previous solution as a starting point
while (tiempo < T):
    #Display report simulation
    from dolfinx import log
    log.set_log_level(log.LogLevel.INFO)
    ##Update time-dependent parameters:
    loaddisp_.t = tiempo
    loaddisp.interpolate(loaddisp_.eval)

    if tiempo < T_ramp:
        t_.value = tiempo
    else:
        t_.value = 0
    
    if tiempo <= T_ramp:
        print('Applying Dirichlet BC')
    else:
        print('Free oscillation')
    
    print("dt: ",dt)
    print("Step: ",steps)
    print("Tiempo: ",tiempo,"/",T)
            
    if tiempo < T_ramp: solver_problem_ugi1.solve(u)
    else: solver_problem_ugi2.solve(u)

    a_vec = dolfinx.fem.Expression(update_a(u,u_old,v_old,a_old), V.element.interpolation_points());
    v_vec = dolfinx.fem.Expression(update_v(update_a(u,u_old,v_old,a_old),u_old,v_old,a_old), V.element.interpolation_points());
    u_vec = dolfinx.fem.Expression(u, V.element.interpolation_points());
    a_old.interpolate(a_vec)
    v_old.interpolate(v_vec)
    u_old.interpolate(u_vec)

    log.set_log_level(log.LogLevel.OFF)
    
    # Write to .xdmf results file
    uVector=dolfinx.fem.Function(VV); uVector.name = "Displacement"; u_expr = dolfinx.fem.Expression(u, VV.element.interpolation_points()); uVector.interpolate(u_expr)
    file_results.write_function(uVector,tiempo)

    FTensor=dolfinx.fem.Function(TT); FTensor.name = "F"; F_expr = dolfinx.fem.Expression(F, TT.element.interpolation_points()); FTensor.interpolate(F_expr)
    file_results.write_function(FTensor,tiempo)

    # EVALUATE DISPLACEMENT AT A POINT (CONTINUATION)#####
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    disp1 = u.eval(points_on_proc, cells)
    disp_x_point1 = np.append(disp_x_point1,disp1[0])
    time_vec = np.append(time_vec,tiempo)
    ######################################################

    np.savetxt("displacement_x_point1.txt", disp_x_point1)
    np.savetxt("time.txt", time_vec)

    plt.figure()
    plt.plot(time_vec[:],disp_x_point1[:])
    plt.grid();plt.xlabel("Time [s]");plt.ylabel("Displacement x [mm]")
    plt.savefig('Oscillation_x_point1.png')
    plt.close()
    
    # Define boundary conditions
    steps += 1
    tiempo += dt
    file_results.close()

print("Finished")
