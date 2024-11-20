import streamlit as st
import numpy as np
import random
import math
import plotly.graph_objects as go
from functools import partial


class ConstrainedOptimizationProblem:
    def __init__(self, objective_function, constraint_function=None, is_feasible=None):
        self.objective_function = objective_function
        self.constraint_function = constraint_function
        self.is_feasible = is_feasible

    def evaluate(self, x, y):
        return self.objective_function(x, y)

    def is_constraint_satisfied(self, x, y):
        if self.is_feasible is None:
            return True
        else:
            return self.is_feasible(x, y)


class SimulatedAnnealing:
    def __init__(
        self,
        problem,
        bounds,
        num_iterations,
        initial_temperature,
        cooling_rate,
        init_solution=None,
    ):
        self.problem = problem
        self.bounds = bounds
        self.num_iterations = num_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.init_solution = init_solution

    def generate_initial_solution(self):
        if self.init_solution is not None:
            return self.init_solution[0], self.init_solution[1]
        while True:
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            if self.problem.is_constraint_satisfied(x, y):
                return x, y

    def generate_new_solution(self, current_x, current_y):
        while True:
            new_x = current_x + random.uniform(-1, 1)
            new_y = current_y + random.uniform(-1, 1)

            # Ensure the new solution is within bounds
            new_x = max(self.bounds[0][0], min(new_x, self.bounds[0][1]))
            new_y = max(self.bounds[1][0], min(new_y, self.bounds[1][1]))

            if self.problem.is_constraint_satisfied(new_x, new_y):
                return new_x, new_y

    def run(self):
        current_x, current_y = self.generate_initial_solution()
        current_cost = self.problem.evaluate(current_x, current_y)

        best_x, best_y = current_x, current_y
        best_cost = current_cost

        temperature = self.initial_temperature

        # Track the trajectory of solutions
        trajectory_x = [current_x]
        trajectory_y = [current_y]

        for _ in range(self.num_iterations):
            new_x, new_y = self.generate_new_solution(current_x, current_y)
            new_cost = self.problem.evaluate(new_x, new_y)

            delta = new_cost - current_cost

            if delta < 0:
                current_x, current_y = new_x, new_y
                current_cost = new_cost

                if new_cost < best_cost:
                    best_x, best_y = new_x, new_y
                    best_cost = new_cost
            else:
                probability = math.exp(-delta / temperature)
                if random.random() < probability:
                    current_x, current_y = new_x, new_y
                    current_cost = new_cost

            temperature *= self.cooling_rate
            temperature = np.max([temperature, 0.0001])

            # Update the trajectory
            trajectory_x.append(current_x)
            trajectory_y.append(current_y)

        return best_x, best_y, best_cost, trajectory_x, trajectory_y


### Define test optimization functions:


def booth_func(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def matyas_func(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


def bukin_func(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


def sample_objective_func(x, y):
    return x**2 + y**2


# Define the objective function
def define_objective(name, x, y):
    if name == "Booth":
        return objective_function(x, y, booth_func)
    elif name == "Matyas":
        return objective_function(x, y, matyas_func)
    elif name == "Bukin":
        return objective_function(x, y, bukin_func)
    elif name == "Circle":
        return objective_function(x, y, sample_objective_func)


def objective_function(x, y, f):
    return f(x, y)


# Define the constraint function (optional)
# def constraint_function(x, y):
#     return x + y - 1


def constraint_function(x, y):
    return x + y


# Check if a solution satisfies the constraint (optional)
def is_feasible(x, y):
    return constraint_function(x, y) >= 0


def main():
    st.set_page_config("Simulated Annealing", layout="wide")
    st.title("Simulated Annealing Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu",
    )
    with st.sidebar:
        selected_objective_func_ = st.selectbox(
            "Select a test function",
            ["Booth", "Matyas", "Bukin", "Circle"],
        )
    objective_function = partial(define_objective, selected_objective_func_)
    # Create an instance of the ConstrainedOptimizationProblem class
    problem_with_constraint = ConstrainedOptimizationProblem(
        objective_function, constraint_function, is_feasible
    )
    # Define the bounds for x and y

    # bounds = [(x_lb, x_ub), (y_lb, y_ub)]
    bounds = [(-10, 10), (-10, 10)]
    # Parameters for Simulated Annealing

    # num_iterations = 1000
    # initial_temperature = 1000
    # cooling_rate = 0.99

    with st.sidebar:

        num_iterations = st.number_input(
            "Number of iterations", min_value=1, max_value=10000, value=100
        )
        initial_temperature = st.number_input(
            "Initial temperature", min_value=100, max_value=10000, value=1000
        )
        cooling_rate = st.number_input(
            "Cooling rate", min_value=0.001, max_value=0.999, value=0.99
        )
        target_obj = st.number_input(
            "Target objective", min_value=-1000, max_value=1000, value=0
        )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                opt_x_loc = st.number_input(
                    "Target x-coordinate", min_value=-10, max_value=10, value=0
                )
            with col2:
                opt_y_loc = st.number_input(
                    "Target y-coordinate", min_value=-10, max_value=10, value=0
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                init_x_loc = st.number_input(
                    "Init x-coordinate", min_value=-10, max_value=10, value=0
                )
            with col2:
                init_y_loc = st.number_input(
                    "Init y-coordinate", min_value=-10, max_value=10, value=0
                )
            randomized_init = st.checkbox("Randomize initial solution")

    # Generate a grid of points for contour plotting
    x_grid = np.linspace(bounds[0][0] - 5, bounds[0][1] + 5, 100)
    y_grid = np.linspace(bounds[1][0] - 5, bounds[1][1] + 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = define_objective(selected_objective_func_, X, Y)
    # Create a contour plot
    fig = go.Figure(
        data=go.Contour(
            x=x_grid,
            y=y_grid,
            z=Z,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Objective Function Value",
                x=1.05,  # Move colorbar to the right
                y=0.5,
                yanchor="middle",
            ),
        )
    )

    # Run Simulated Annealing with constraint
    if not randomized_init:
        init_solution = (init_x_loc, init_y_loc)
    else:
        init_solution = None

    sa_with_constraint = SimulatedAnnealing(
        problem_with_constraint,
        bounds,
        num_iterations,
        initial_temperature,
        cooling_rate,
        init_solution,
    )
    (
        best_x_with_constraint,
        best_y_with_constraint,
        best_cost_with_constraint,
        trajectory_x_with_constraint,
        trajectory_y_with_constraint,
    ) = sa_with_constraint.run()
    with st.sidebar:
        st.markdown("# Solution")
        with st.container():
            col1, col2 = st.columns(2)
            prec_ = 3
            with col1:
                x_diff_from_target = best_x_with_constraint - opt_x_loc
                st.metric(
                    label="Best x-solution",
                    value=np.round(best_x_with_constraint, prec_),
                    delta=float(np.round(x_diff_from_target, prec_)),
                )
            with col2:
                y_diff_from_target = best_y_with_constraint - opt_y_loc
                st.metric(
                    label="Best y-solution",
                    value=np.round(best_y_with_constraint, prec_),
                    delta=float(np.round(y_diff_from_target, prec_)),
                )
        obj_diff_from_target = best_cost_with_constraint - target_obj
        st.metric(
            label="Objective Value",
            value=np.round(best_cost_with_constraint, prec_),
            delta=float(np.round(obj_diff_from_target, prec_)),
        )
    # Plot the entire trajectory including the initial point
    fig.add_trace(
        go.Scatter(
            x=trajectory_x_with_constraint,
            y=trajectory_y_with_constraint,
            mode="lines+markers",
            line=dict(color="blue", width=0.5),
            marker=dict(color="blue", symbol="x", size=5),
            name="Solution Trajectory",
            showlegend=True,
        )
    )

    # Highlight the initial solution with a different marker
    fig.add_trace(
        go.Scatter(
            x=[trajectory_x_with_constraint[0]],
            y=[trajectory_y_with_constraint[0]],
            mode="markers",
            marker=dict(color="red", symbol="circle", size=10),
            name="Initial Solution",
            showlegend=True,
        )
    )

    # Plot the best solution found by Simulated Annealing
    fig.add_trace(
        go.Scatter(
            x=[best_x_with_constraint],
            y=[best_y_with_constraint],
            mode="markers",
            marker=dict(color="green", symbol="square", size=10),
            name="Best Solution",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[opt_x_loc],
            y=[opt_y_loc],
            mode="markers",
            marker=dict(color="yellow", symbol="star", size=10),
            name="Optimal Solution",
            showlegend=True,
        )
    )

    # Adjust the legend positions
    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        ),
        margin=dict(
            l=50, r=200, t=50, b=50
        ),  # Increase right margin to accommodate colorbar
    )

    fig.update_layout(xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
