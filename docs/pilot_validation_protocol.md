# Pilot Validation Protocol

## Goal
Provide a pilot-ready validation package for the cognitive caregiving robot without fabricating field results.

## Scope
This protocol prepares the repository for assisted-living or hospital-like pilot deployment by defining:
- inclusion criteria
- sensing profiles
- override policies
- risk escalation workflow
- caregiver interaction logging
- outcome and safety endpoints

## Proposed Pilot Stages
1. Dry-run digital-twin replay with caregiver observers.
2. Controlled room deployment without autonomous intervention.
3. Advisory-mode deployment with dashboard review for all alerts.
4. Limited telepresence escalation with clinician supervision.

## Primary Endpoints
- alert acknowledgment latency
- false escalation rate
- adherence intervention usefulness
- explanation usefulness and faithfulness
- user and caregiver acceptability

## Safety Gates
- all high-risk outputs require human acknowledgment
- medication actions remain advisory unless externally approved
- privacy profile must be documented per participant
- all telepresence sessions require audit logging

## External Dependencies
The following remain external to the repository and must be completed before a real pilot:
- ethics and consent approval
- site agreement
- hardware and network provisioning
- participant recruitment
- clinician review board

## Repository Support
The repository now includes:
- benchmark and figure pipeline
- dashboard prototype
- populated care knowledge graph
- pilot readiness table
- simulated alert and physiology streams

These support pilot preparation, but they do not constitute completed real-world validation.
