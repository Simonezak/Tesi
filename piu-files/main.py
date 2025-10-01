import wntr

wn = wntr.network.WaterNetworkModel(r"C:\Users\nephr\Desktop\Uni-Nuova\Tesi\WNTR-main\WNTR-main\examples\networks\Net3.inp")
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

print("Time steps disponibili (in secondi):")
print(results.time)        # array di tempi (in secondi)
print("Numero timestep:", len(results.time))
