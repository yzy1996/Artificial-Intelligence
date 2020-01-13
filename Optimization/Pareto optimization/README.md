for the conflicts between different objectives



Existing approaches for Pareto optimization can be categorized into two categories: **heuristic search** and **scalarization**



**Evolutionary algorithms** are popular choices in **heuristic search** approaches



However, **heuristic search** can not guarantee **Pareto efficiency**, it only ensures the resulting solutions are not dominated by each other (but still can be dominated by the Pareto efficient solutions) [Eckart Zitzler, Marco Laumanns, and Lothar Thiele. 2001. SPEA2: Improving the
Strength Pareto Evolutionary Algorithm. Technical Report]



**Scalarization** transforms multiple objectives into a single one with a **weighted sum** of all the objective functions



However, the **scalarization weights** of objective functions are usually determined **manually** and Pareto efficiency is still not guaranteed

(KKT) conditions can be used to guide the scalarization

Multiple-Gradient Descent Algorithm ( MGDA ).