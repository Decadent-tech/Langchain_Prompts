from pydantic import BaseModel, EmailStr,Field
from typing import Optional


class Student(BaseModel):
    name: str = None
    age: int
    address: Optional[str] = None
    email:EmailStr
    cgpa:float = Field(gt=0.0, lt=10.0,default=5.0,description="CGPA must be between 0.0 and 10.0")

new_student = {'age':'32','email':'abc@yahoo.co','cgpa':8.5}
student = Student(**new_student)
print(dict(student))
print(student.email)