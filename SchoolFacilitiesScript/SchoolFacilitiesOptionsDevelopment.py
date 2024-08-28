import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class SchoolFacility:
    def __init__(self, name, condition, capacity, current_enrollment, age):
        self.name = name
        self.condition = condition  # 0-100 scale
        self.capacity = capacity
        self.current_enrollment = current_enrollment
        self.age = age

class SchoolDistrict:
    def __init__(self, name):
        self.name = name
        self.schools = []
        self.population_density = 0
        self.bond_amount = 0

class Project:
    def __init__(self, name, school, cost, duration, impact):
        self.name = name
        self.school = school
        self.cost = cost
        self.duration = duration  # in months
        self.impact = impact  # description of the impact

class BondFundAllocationSystem:
    def __init__(self):
        self.district = None
        self.enrollment_model = None

    def load_district_data(self, district):
        self.district = district

    def predict_future_enrollment(self, years_ahead=5):
        X = pd.DataFrame([(school.age, school.capacity, school.condition) for school in self.district.schools],
                         columns=['age', 'capacity', 'condition'])
        y = pd.Series([school.current_enrollment for school in self.district.schools])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.enrollment_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.enrollment_model.fit(X_train, y_train)

        future_enrollments = []
        for school in self.district.schools:
            future_age = school.age + years_ahead
            future_condition = max(0, school.condition - (2 * years_ahead))
            future_enrollment = self.enrollment_model.predict([[future_age, school.capacity, future_condition]])[0]
            future_enrollments.append(future_enrollment)

        return future_enrollments

    def cluster_schools(self):
        X = pd.DataFrame([(school.condition, school.capacity, school.current_enrollment, school.age) 
                          for school in self.district.schools],
                         columns=['condition', 'capacity', 'enrollment', 'age'])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        return clusters

    def generate_projects(self):
        future_enrollments = self.predict_future_enrollment()
        clusters = self.cluster_schools()

        projects = []

        for i, school in enumerate(self.district.schools):
            current_utilization = school.current_enrollment / school.capacity
            future_utilization = future_enrollments[i] / school.capacity
            cluster = clusters[i]

            if cluster == 0:  # Good condition cluster
                if future_utilization > 0.9:
                    projects.append(Project(f"Expand {school.name}", school, 
                                            cost=10000000, duration=18, 
                                            impact="Increase capacity to meet future enrollment"))
            elif cluster == 1:  # Average condition cluster
                if future_utilization > 0.85:
                    projects.append(Project(f"Renovate and expand {school.name}", school, 
                                            cost=15000000, duration=24, 
                                            impact="Modernize facilities and increase capacity"))
                else:
                    projects.append(Project(f"Modernize {school.name}", school, 
                                            cost=8000000, duration=12, 
                                            impact="Upgrade facilities to improve learning environment"))
            else:  # Poor condition cluster
                if future_utilization > 0.8:
                    projects.append(Project(f"Rebuild {school.name}", school, 
                                            cost=25000000, duration=36, 
                                            impact="Replace outdated facility with modern, larger school"))
                else:
                    projects.append(Project(f"Major renovation of {school.name}", school, 
                                            cost=20000000, duration=24, 
                                            impact="Comprehensively update and repair facility"))

        # Evaluate need for new school
        total_future_enrollment = sum(future_enrollments)
        total_capacity = sum(school.capacity for school in self.district.schools)
        if total_future_enrollment > 0.95 * total_capacity:
            projects.append(Project("Construct new school", None, 
                                    cost=40000000, duration=48, 
                                    impact="Add capacity to accommodate district-wide growth"))

        return projects

    def allocate_bond_funds(self, projects):
        # Sort projects by a priority score (cost-effectiveness)
        for project in projects:
            if project.school:
                enrollment_impact = (project.school.capacity * 0.1) if "expand" in project.name.lower() else 0
                condition_impact = 20 if "rebuild" in project.name.lower() else 10 if "renovate" in project.name.lower() else 5
            else:
                enrollment_impact = 500  # Assumed impact for a new school
                condition_impact = 30
            project.priority_score = (enrollment_impact + condition_impact) / project.cost

        sorted_projects = sorted(projects, key=lambda x: x.priority_score, reverse=True)

        allocated_projects = []
        remaining_funds = self.district.bond_amount
        total_duration = 0

        for project in sorted_projects:
            if project.cost <= remaining_funds:
                allocated_projects.append(project)
                remaining_funds -= project.cost
                total_duration = max(total_duration, project.duration)

        return allocated_projects, total_duration

def main():
    # Create a sample school district
    district = SchoolDistrict("Sample Unified School District")
    district.schools = [
        SchoolFacility("High School A", 75, 1500, 1300, 25),
        SchoolFacility("Middle School B", 60, 1000, 850, 35),
        SchoolFacility("Elementary School C", 90, 600, 580, 10),
        SchoolFacility("Elementary School D", 40, 500, 300, 50),
    ]
    district.population_density = 3000
    district.bond_amount = 100000000  # $100 million bond

    # Initialize and run the bond fund allocation system
    bfas = BondFundAllocationSystem()
    bfas.load_district_data(district)
    
    all_projects = bfas.generate_projects()
    allocated_projects, total_duration = bfas.allocate_bond_funds(all_projects)

    # Print results
    print(f"Bond Fund Allocation Plan for {district.name}")
    print(f"Total Bond Amount: ${district.bond_amount:,}")
    print(f"\nProposed Projects:")
    total_cost = 0
    for project in allocated_projects:
        print(f"- {project.name}")
        print(f"  Cost: ${project.cost:,}")
        print(f"  Duration: {project.duration} months")
        print(f"  Impact: {project.impact}")
        print()
        total_cost += project.cost

    print(f"Total Allocated Funds: ${total_cost:,}")
    print(f"Remaining Funds: ${district.bond_amount - total_cost:,}")
    print(f"\nOverall Project Timeline: {total_duration} months")

    # List impacted schools
    impacted_schools = set(project.school.name for project in allocated_projects if project.school)
    print("\nImpacted Schools:")
    for school in impacted_schools:
        print(f"- {school}")

    if any(project.name == "Construct new school" for project in allocated_projects):
        print("- New School to be constructed")

if __name__ == "__main__":
    main()
