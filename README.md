# IntelligentSystemsProject
The overarching project comprises four submodules: autonomous farming robots, crop health monitoring, crop health management, and the AI-powered central control unit. Given the limited 3-week time frame for implementing the proof of concept (POC) for one of these submodules, and the hardware-intensive nature of most of them, the primary emphasis has been placed on the machine learning aspects, particularly those associated with crop health management and the AI-powered central control unit.

As outlined in the previous report, the central control unit is a substantial submodule requiring the formulation of a policy for the pretraining of a model based on farmers' knowledge, historical practices, and weather forecast data. Furthermore, the initial trained model must undergo improvement through reinforcement learning, necessitating the development of a comprehensive environment. Consequently, the POC to be executed in the next 3 weeks is directed towards a more manageable submodule.

Numerous research papers affirm that crop health monitoring is achievable through multispectral images, with publicly available datasets supporting this claim. As detailed in the comprehensive report, two AI models will be developed to showcase the concept of crop health monitoring using drones. The process involves drones capturing multispectral images of a large area, identifying areas of interest (such as stressed crops). Subsequently, a more precise image is transmitted to the control unit to discern whether the crop lacks water/nutrients or is afflicted by a disease.

Given the time constraints and the limited availability of publicly accessible data on water or nutrient deficiency using multispectral cameras, this POC will solely focus on disease identification. 

This GitHub contains the public dataset used and the implementation.
