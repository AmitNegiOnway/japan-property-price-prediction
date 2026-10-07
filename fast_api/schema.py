from pydantic import BaseModel , Field
from typing import Annotated , List

class InputSchema(BaseModel):
    MunicipalityCode: Annotated[int,Field(title="MunicipalityCode",description="MunicipalityCode of that area")]
       
    MinTimeToNearestStation: Annotated[int,Field(ge=0,le=120,title="MinTimeToNearestStation",description="Distance of property location to nearest metro station in minutes")]
    
    Area: Annotated[int,Field(ge=15,le=5000, title="Area", description="Area of house in square meters")]
    
    Frontage: Annotated[int,Field(ge=0,le=50,title="Frontage",description="Frontage of the property in meters")]
    
    TotalFloorArea: Annotated[int,Field(ge=10,le=2000, title="TotalFloorArea",description="Total floor area in square meters")]
    
    BuildingYear: Annotated[int,Field(ge=1945,le=2020,title="BuildingYear",description="Year the building was constructed")]
    
    Breadth: Annotated[int,Field(ge=1,le=60,title="Breadth",description="Breadth of the property in meters")]
    
    CoverageRatio: Annotated[int,Field(ge=30,le=80,title="CoverageRatio",description="Building coverage ratio (percentage)")]
    
    FloorAreaRatio: Annotated[int,Field(ge=50,le=800,title="FloorAreaRatio",description="Floor area ratio (percentage)")]
    
    Year: Annotated[int,Field(ge=2006,le=2019,title="Year",description="Year of the transaction or record")]
    
    Quarter: Annotated[int,Field(ge=1,le=4,title="Quarter",description="Quarter of the year (1–4)")]

    Type: Annotated[str,Field(title="Type",description="Type of the property or land")]
    
    Region: Annotated[str,Field(title="Region",description="Geographical region where the property is located")]
    
    DistrictName: Annotated[str,Field(title="District Name",description="Name of the district where the property is located")]
    
    NearestStation: Annotated[str,Field(title="Nearest Station",description="Name of the nearest station")]
    
    LandShape: Annotated[str,Field(title="Land Shape",description="Shape or configuration of the land")]
    
    Structure: Annotated[str,Field(title="Structure",description="Structural type of the property")]
    
    Use: Annotated[str,Field(title="Use",description="Primary use of the property")]
    
    Purpose: Annotated[str,Field(title="Purpose",description="Purpose of the property")]
    
    Direction: Annotated[str,Field(title="Direction",description="Direction or orientation of the property")]
    
    Classification: Annotated[str,Field(title="Classification",description="Classification of the property")]
    
    CityPlanning: Annotated[str,Field(title="City Planning",description="City planning classification of the property")]


class OutputSchema(BaseModel):
    Price:str = Field(...,description='The Predicted Price of the house in Japan ')





