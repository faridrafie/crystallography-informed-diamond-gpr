# Project information

**Project:** Crystallography-Informed Gaussian-Process Surrogate Modeling of Low-Energy Nitrogen Implantation in Diamond  
**Researcher / code maintainer:** Seyed Faridedin Rafie (Farid Rafie)  
**Affiliation:** Materials Science and Engineering Department, University of Wisconsin-Milwaukee, Milwaukee, WI 53211, USA  
**Research period:** 2025–2026

## Research context

This codebase was organized as part of Farid Rafie's doctoral research workflow on atomistic modeling and data-driven design of nitrogen implantation in diamond. The public package is intentionally focused on the fixed-azimuth study reported in the manuscript.

## Reproducibility principle

The repository contains the processed numerical tables required by the manuscript-level surrogate analyses, the frozen crystallographic representation, validation outputs, numerical figure source data, and analysis scripts. Large raw MD trajectory archives are maintained separately because they are not required to reproduce the reported surrogate calculations from the processed data.

## Coordinate convention

`z || [001]`, `x || [100]`, and `y || [010]`; polar tilt `θ` is measured from `[001]`, and azimuth `φ` from `[100]` toward `[010]`. The primary manuscript campaign uses `φ = 0°`.

## Authorship note

The Python-file headers identify the researcher and maintainer of this research codebase. They provide project provenance and do not replace the authorship/contribution statement of the associated manuscript.
