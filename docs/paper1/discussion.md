# Discussion

## What Worked
The repository now supports a clean Paper 1 execution path with CS1, CS2, and CS3 isolated from later-paper functionality. The existing baseline was preserved rather than replaced, which keeps a traceable connection to the original project. The CSV-first design also makes the codebase easy to translate into manuscript tables and figures.

## Limitations
The strongest limitation is data realism. CS1 and CS2 use software-equivalent or synthetic placeholder streams, and CS3 preserves only a real visual baseline while using synthetic placeholder multimodal data for B1 through B3. The current label space also reflects actual repository evidence, which is `happy/sad/neutral/fear` rather than the originally intended `angry` class.

## Deployment Implications
Despite those limitations, the repository now provides the right backbone for a publishable first paper: measurable interfaces, synchronization reporting, fault studies, modality tracking, baseline preservation, and benchmark automation. This makes future simulator integration, ROS2 deployment, and site data collection substantially easier.

## Why This Supports Later Papers
Paper 1 is intentionally infrastructural. By making the digital twin, synchronized windows, and benchmark stack measurable now, later modules such as HITL control, medication adherence, explainability, and privacy-aware deployment can be evaluated on top of a stable experimental substrate.
