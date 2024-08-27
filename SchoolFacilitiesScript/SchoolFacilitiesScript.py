import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class SchoolDistrict:
    def __init__(self, name):
        self.name = name
        self.schools = []
        self.maintenance_tickets = []

class School:
    def __init__(self, name, school_type):
        self.name = name
        self.type = school_type
        self.buildings = []
        self.staff = []
        self.students = []

class Building:
    def __init__(self, name):
        self.name = name
        self.rooms = []
        self.equipment = []

class Room:
    def __init__(self, name, capacity, room_type):
        self.name = name
        self.capacity = capacity
        self.type = room_type
        self.utilization = 0
        self.schedule = {day: {hour: None for hour in range(8, 17)} for day in range(5)}

class Equipment:
    def __init__(self, name, installation_date, expected_lifespan):
        self.name = name
        self.installation_date = installation_date
        self.last_maintenance = installation_date
        self.condition = 100
        self.expected_lifespan = expected_lifespan

class Person:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.schedule = {day: {hour: None for hour in range(8, 17)} for day in range(5)}

class MaintenanceTicket:
    def __init__(self, school, building, equipment, description, resolution=None):
        self.school = school
        self.building = building
        self.equipment = equipment
        self.description = description
        self.resolution = resolution
        self.date_submitted = datetime.now()
        self.date_resolved = None

def generate_sample_district():
    district = SchoolDistrict("Sample Unified School District")
    school_types = ["Elementary", "Middle", "High"]
    
    for school_type in school_types:
        for i in range(random.randint(3, 5)):
            school = School(f"{school_type} School {i+1}", school_type)
            
            for j in range(random.randint(1, 3)):
                building = Building(f"Building {j+1}")
                
                room_types = ["Classroom", "Lab", "Gym", "Cafeteria", "Library", "Office"]
                for k in range(random.randint(20, 50)):
                    room_type = random.choice(room_types)
                    capacity = random.randint(20, 35) if room_type == "Classroom" else random.randint(50, 200)
                    building.rooms.append(Room(f"{room_type} {k+1}", capacity, room_type))
                
                equipment_types = ["HVAC", "Projector", "Computer", "Smartboard", "Printer"]
                for _ in range(random.randint(10, 30)):
                    equip_type = random.choice(equipment_types)
                    install_date = datetime.now() - timedelta(days=random.randint(365, 3650))
                    lifespan = random.randint(5, 15)
                    building.equipment.append(Equipment(equip_type, install_date, lifespan))
                
                school.buildings.append(building)
            
            num_students = random.randint(300, 1000)
            num_teachers = num_students // 20
            
            for _ in range(num_teachers):
                school.staff.append(Person(f"Teacher {_}", "Teacher"))
            
            for _ in range(num_students):
                school.students.append(Person(f"Student {_}", "Student"))
            
            for _ in range(random.randint(3, 7)):
                school.staff.append(Person(f"Admin {_}", "Administrator"))
            
            district.schools.append(school)
    
    return district

def generate_sample_maintenance_tickets(district):
    issues = [
        "not working", "broken", "malfunctioning", "needs repair",
        "strange noise", "overheating", "leaking", "outdated",
        "slow performance", "error messages"
    ]
    resolutions = [
        "replaced part", "software update", "cleaned thoroughly",
        "adjusted settings", "repaired wiring", "lubricated moving parts",
        "recalibrated", "reinstalled", "upgraded components", "scheduled for replacement"
    ]
    
    for _ in range(1000):  # Generate 1000 sample tickets
        school = random.choice(district.schools)
        building = random.choice(school.buildings)
        equipment = random.choice(building.equipment)
        issue = random.choice(issues)
        resolution = random.choice(resolutions) if random.random() > 0.2 else None  # 80% resolved
        
        description = f"{equipment.name} is {issue} in {building.name} of {school.name}"
        ticket = MaintenanceTicket(school, building, equipment, description, resolution)
        
        if resolution:
            ticket.date_resolved = ticket.date_submitted + timedelta(days=random.randint(1, 30))
        
        district.maintenance_tickets.append(ticket)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if not w in stop_words and w.isalnum()])

def train_maintenance_classifier(district):
    resolved_tickets = [ticket for ticket in district.maintenance_tickets if ticket.resolution]
    descriptions = [preprocess_text(ticket.description) for ticket in resolved_tickets]
    resolutions = [preprocess_text(ticket.resolution) for ticket in resolved_tickets]
    
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(descriptions)
    y = resolutions
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    accuracy = classifier.score(X_test, y_test)
    print(f"Maintenance Classifier Accuracy: {accuracy:.2f}")
    
    return vectorizer, classifier

def predict_maintenance(district, vectorizer, classifier):
    print("\nPredictive Maintenance Report")
    print("==============================")
    
    current_date = datetime.now()
    urgent_maintenance = []
    upcoming_maintenance = []
    
    for school in district.schools:
        for building in school.buildings:
            for equipment in building.equipment:
                age = (current_date - equipment.installation_date).days / 365
                time_since_maintenance = (current_date - equipment.last_maintenance).days
                
                equipment.condition -= random.uniform(0, 10) * age / equipment.expected_lifespan
                equipment.condition -= random.uniform(0, 5) * time_since_maintenance / 180
                equipment.condition = max(0, equipment.condition)
                
                if equipment.condition < 60:
                    urgent_maintenance.append((school, building, equipment))
                elif equipment.condition < 80:
                    upcoming_maintenance.append((school, building, equipment))
    
    print("Urgent Maintenance Required:")
    for school, building, equipment in urgent_maintenance[:10]:
        description = f"{equipment.name} requires maintenance in {building.name} of {school.name}"
        processed_description = preprocess_text(description)
        predicted_resolution = classifier.predict(vectorizer.transform([processed_description]))[0]
        print(f"- {description}")
        print(f"  Predicted Resolution: {predicted_resolution}")
    
    print("\nUpcoming Maintenance:")
    for school, building, equipment in upcoming_maintenance[:10]:
        description = f"{equipment.name} may need maintenance soon in {building.name} of {school.name}"
        processed_description = preprocess_text(description)
        predicted_resolution = classifier.predict(vectorizer.transform([processed_description]))[0]
        print(f"- {description}")
        print(f"  Predicted Resolution: {predicted_resolution}")

def analyze_maintenance_trends(district):
    print("\nMaintenance Trend Analysis")
    print("===========================")
    
    equipment_issues = {}
    for ticket in district.maintenance_tickets:
        if ticket.equipment.name not in equipment_issues:
            equipment_issues[ticket.equipment.name] = 0
        equipment_issues[ticket.equipment.name] += 1
    
    sorted_issues = sorted(equipment_issues.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 Equipment with Most Issues:")
    for equipment, count in sorted_issues[:5]:
        print(f"- {equipment}: {count} issues")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar([item[0] for item in sorted_issues[:10]], [item[1] for item in sorted_issues[:10]])
    plt.title("Top 10 Equipment Types by Number of Issues")
    plt.xlabel("Equipment Type")
    plt.ylabel("Number of Issues")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    district = generate_sample_district()
    generate_sample_maintenance_tickets(district)
    
    vectorizer, classifier = train_maintenance_classifier(district)
    predict_maintenance(district, vectorizer, classifier)
    analyze_maintenance_trends(district)

if __name__ == "__main__":
    main()