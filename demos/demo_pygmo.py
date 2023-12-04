

# Import pygmo
import pygmo as pg

# # Create the optimization algorithm
# myAlgorithm = pg.algorithm(pg.nlopt(algorithm_name))
# myAlgorithm.extract(pg.nlopt).xtol_rel = 1e-6
# myAlgorithm.extract(pg.nlopt).ftol_rel = 1e-6
# myAlgorithm.extract(pg.nlopt).xtol_abs = 1e-6
# myAlgorithm.extract(pg.nlopt).ftol_abs = 1e-6
# myAlgorithm.extract(pg.nlopt).maxeval = 100
# myAlgorithm.set_verbosity(0)s

# # Create the optimization problem
# myProblem = pg.problem(self.PointToCurveProjectionProblem(self.get_value, self.get_derivative, P))

# # Create the population
# myPopulation = pg.population(prob=myProblem, size=0)

# # Create a list with the different starting points
# U0 = self.U[0:-1] + 1/2 * (self.U[1:] - self.U[0:-1])
# for u0 in U0:
#     myPopulation.push_back([u0])

# # Solve the optimization problem (evolve the population in Pygmo's jargon)
# myPopulation = myAlgorithm.evolve(myPopulation)

# # Get the optimum
# u = myPopulation.champion_x[0]

# return u

# class PointToCurveProjectionProblem:

# def __init__(self, C, dC, P):
#     """ Solve point inversion problem: min(u) ||C(u) - P|| """
#     self.C_func = C
#     self.dC_func = dC
#     self.P = np.reshape(P, (P.shape[0], 1))

# @staticmethod
# def get_bounds():
#     """ Set the bounds for the optimization problem """
#     return [0.00], [1.00]

# def fitness(self, x):
#     """ Evaluate the deviation between the prescribed point and the parametrized point """
#     u = np.asarray([x[0]])
#     C = self.C_func(u)
#     P = self.P
#     return np.asarray([np.sum(np.sum((C - P) ** 2, axis=0) ** (1 / 2))])

# def gradient(self, x):
#     """ Compute the gradient of the fitness function analytically """
#     u = np.asarray([x[0]])
#     C = self.C_func(u)
#     dC = self.dC_func(u, order=1)
#     P = self.P
#     numerator = np.sum((C - P) * dC, axis=0)
#     denominator = np.sum(np.sum((C - P) ** 2, axis=0) ** (1 / 2))
#     if np.abs(denominator) > 0:
#         gradient = numerator/denominator
#     else:
#         gradient = np.asarray(0)[np.newaxis]
#     return gradient
