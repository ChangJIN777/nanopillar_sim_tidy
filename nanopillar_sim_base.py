### Import functions
import sys, os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tidy3d.plugins.resonance import ResonanceFinder
from tidy3d import web
import gdspy
from scipy.integrate import trapz
from scipy.constants import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

import tidy3d as td
  
from phidl import Device, CrossSection
from phidl import quickplot as qp
import phidl.geometry as pg
import phidl.path as pp

import pdb # for debugging the python scripts

# material parameters 
n_Si = 3.9595
n_SiO2 = 1.4585
Si = td.Medium(permittivity=n_Si**2)
SiO2 = td.Medium(permittivity=n_SiO2**2)
air = td.Medium(permittivity=1)

# 1 nanometer in units of microns (for conversion)
nm = 1e-3

class nanopillar():

    def __init__(self):
        # define the nanopillar geometries
        # resonant wavelength of the T center 
        self.res_wavelength = 1326*nm 
        self.f0 = td.C_0/self.res_wavelength 
        self.fwidth = 20e12
        self.num_freqs = 100
        self.run_time = 3e-12 # simulation runtime in seconds
        self.freqs = np.linspace(self.f0-self.fwidth/2,self.f0+self.fwidth/2,self.num_freqs)

        # shape parameters of the pillar 
        self.H = 220 * nm #um
        self.W = 300 * nm #um
        self.bulk_thickness = 10 #um
        self.bulk_Si_thickness = 110 * nm #um

        # location of the dipole
        self.defect_z = 110 * nm #um

        # location of the monitor 
        self.top_monitor_z = 1000 * nm # the z location of the top flux monitor 
        self.bottom_monitor_z = -1000 * nm # the z location of the bottom flux monitor  

        # the size of the simluation 
        self.sim_size = [2,2,4]

        # data file destination 
        self.data_path = './data/'
        self.file_name = 'test_data.hdf5'
        self.task_name = 'nanopillar_testRune'

        # the size of the mesh box (for specifying the refined region of meshing)
        self.meshBox_x = self.W*2
        self.meshBox_y = self.W*2 
        self.meshBox_z = self.H*2

        # t start for apodization 
        self.t_start = 5e-13


    def setup_sim(self):
        # define the substrate
        bulk_substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, -self.bulk_thickness/2],
                size=[td.inf, td.inf, self.bulk_thickness],
            ),
            medium=SiO2
        )

        # define the nanopillar structure 
        nanopillar_substrate = td.Structure(
            geometry=td.Cylinder(
                center=(0, 0, self.H/2),
                radius=self.W/2,
                length=self.H,
                axis=2
            ),
            medium=Si
        )

        # define the bulk Si structure 
        bulk_Si = td.Structure(
            geometry=td.Box(
                center=[0, 0, self.bulk_Si_thickness/2],
                size=[td.inf, td.inf, self.bulk_Si_thickness],
            ),
            medium=Si
        )


        # define the dipole source
        pulse = td.GaussianPulse(freq0=self.f0,fwidth=20e12)
        freqs = np.linspace(self.f0-self.fwidth/2,self.f0+self.fwidth/2,self.num_freqs)
        source = td.PointDipole(center=(0,0,self.H-self.defect_z), source_time=pulse, polarization='Ex')

        # Apodization to exclude the source pulse from the frequency-domain monitors
        apodization = td.ApodizationSpec(start=self.t_start, width=2e-13)


        # define the monitors 
        field_monitor = td.FieldMonitor(
                    fields=['Ex'],
                    size=(td.inf, 0, td.inf),
                    center=(0,0,0),
                    freqs=[self.f0],
                    name='field',
                    apodization=apodization)

        flux_monitor_top = td.FluxMonitor(
                    size=(td.inf,td.inf,0),
                    center=(0,0,self.top_monitor_z),
                    freqs=freqs,
                    name='top_monitor',
                    normal_dir = "-")

        flux_monitor_bottom = td.FluxMonitor(
                    size=(td.inf,td.inf,0),
                    center=(0,0,self.bottom_monitor_z),
                    freqs=freqs,
                    name='bottom_monitor',
                    normal_dir = "+")
        
        # setup the mesh box 
        grid_spec = self.getGridSpace(z_center=self.H/2,size_x=self.meshBox_x,size_y=self.meshBox_y,size_z=self.meshBox_z,minStepsPerWl=10)

        nanopillar_sim = td.Simulation(size=self.sim_size,
                               grid_spec=grid_spec,
                               run_time=self.run_time,
                               boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
                               structures = [nanopillar_substrate,bulk_substrate,bulk_Si],
                               sources=[source],
                               monitors = [field_monitor,flux_monitor_top,flux_monitor_bottom])

        # plot the eps at y=0 
        nanopillar_sim.plot_eps(y=0)

        return nanopillar_sim

    def web_runSim(self,sim):
        """this function takes in the tidy3d simulation and submit it to the web server for execution

                    Args:
                        sim (tidy3d simulation object): 
                        cp (cavity parameters dict): 
                        sim_p (simulation paramters dict): 

                    Returns:
                        _type_: _description_
        """
        complete_path = self.data_path + self.file_name
        # Upload task
        task_id = web.upload(sim, task_name = self.task_name)
        # Run simulation
        sim_data = web.run(sim, self.task_name, folder_name = self.data_path, path=complete_path, verbose="True")
        return sim_data

    def plot_field_data(self,sim_data):
        """this function take in the tidy3d simulation result and visualize the data collected by the field monitor

            Args:
                    sim_data (tidy3d simulation result object): 
        """
        sim_data.plot_field("field", "Ex", val="abs", y=0)

    def plot_flux_data(self,sim_data):
        """this function take in the tidy3d simulation result and visualize the data collected by the flux monitor"""
        top_monitor = sim_data['top_monitor'].flux # flux taken in by the top monitor  
        bottom_monitor = sim_data['bottom_monitor'].flux # flux taken in by the bottom monitor  

        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(self.freqs/(1e12),np.abs(top_monitor),label='top monitor')
        ax2.plot(self.freqs/(1e12),np.abs(bottom_monitor),label='bottom monitor')
        ax1.set_ylabel('flux (a.u.)')
        ax2.set_ylabel('flux (a.u.)')
        ax2.set_xlabel('frequency (THz)')
        ax1.legend()
        ax2.legend()
        plt.show()
        return np.abs(top_monitor), np.abs(bottom_monitor)

    def extimate_cost(self,nanopillar_sim):
        verbose = "True"
        job = web.Job(simulation=nanopillar_sim,task_name="nanopillar_job",verbose=verbose)
        estimated_cost = web.estimate_cost(job.task_id)
        print(f'The estimated maximum cost is {estimated_cost:.3f} Flex Credits.')

    def set_pillar_diameter(self,new_diameter):
        self.W = new_diameter

    def set_dataFileName(self,new_dataFileName):
        self.file_name = new_dataFileName

    def getGridSpace(self,x_center=0,y_center=0,z_center=0,size_x = 4,size_y = 0.6,size_z = 0.8,
                        dlX = 0.01,dlY = 0.01,dlZ = 0.01,
                        minStepsPerWl = 10):
            """_summary_

            Args:
                size_x (int, optional): the size of the meshbox in x. Defaults to 4.
                size_y (float, optional): the size of the meshbox in y. Defaults to 0.6.
                size_z (float, optional): the size of the meshbox in z. Defaults to 0.8.
                dlX (float, optional): the resolution of the meshbox in x. Defaults to 0.01.
                dlY (float, optional): the resolution of the meshbox in y. Defaults to 0.01.
                dlZ (float, optional): the resolution of the meshbox in z. Defaults to 0.01.
                minStepsPerWl (int, optional): _description_. Defaults to 10.

            Returns:
                _type_: _description_
            """
            #create a volume object to override the mesh
            geometry = td.Box(center = (x_center,y_center,z_center),
                            size = (size_x,size_y,size_z))
            refineBox = td.MeshOverrideStructure(geometry = geometry,
                                            dl = [dlX,dlY,dlZ])
            return td.GridSpec.auto(min_steps_per_wvl = minStepsPerWl,
                                override_structures = [refineBox])

    def set_t_start(self,t_start):
        self.t_start = t_start
    
    def set_runTime(self,runTime):
        self.run_time = runTime
    
    def sweep_pillar_width(self,pillar_width):
        self.W = pillar_width*nm
        self.set_dataFileName('nanopillar_diameter_%.0fnm.hdf5'%(pillar_width))
        sim = self.setup_sim()
        return sim

    def data_analysis(self,sim_data):
        # save the numpy array as a csv file 
        top_monitor, bottom_monitor = self.plot_flux_data(sim_data)
        data_summary = np.array([self.freqs,
                                    top_monitor,
                                    bottom_monitor])
        data_summary = data_summary.T
        full_path_name = self.data_path + 'nanopillar_diameter_%.0fnm.cvs'%(self.W)
        # adding the headline 
        headline = "freq,top_monitor,bottom_monitor"
        rows = ["{},{},{}".format(i, j, k) for i, j, k in data_summary]
        text = "\n".join(rows)
        text = headline + "\n" + text
        with open(full_path_name,'w') as f:
            f.write(text)
