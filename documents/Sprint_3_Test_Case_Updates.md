# Sprint Test Case Updates – BL-12

## Web-Based ML Model Evaluator

**Backlog Item:** BL-12 – Update Test Cases for Sprint Changes  
**Sprint:** Sprint 3  
**Project:** Web-Based ML Model Evaluator  
**Status:** Testing Completed

---

## 1. Objective

The objective of this task is to review and update the existing test cases according to the latest changes introduced in the project.

The updates cover authentication, user roles, account management, data privacy, model evaluation, visualizations, evaluation history, comparison and PDF report generation.

---

## 2. Scope of Test Case Updates

The following functionalities are covered:

1. User registration and login.
2. Admin and Analyst role management.
3. Account suspension and reactivation.
4. User data isolation and security.
5. Model and dataset validation.
6. Model evaluation and performance metrics.
7. Evaluation history and comparison.
8. Confusion matrix and ROC curve.
9. Regression visualizations.
10. PDF report generation.
11. CSRF protection and error handling.

---

## 3. Updated Test Cases

| Test ID | Feature | Test Scenario | Expected Result |
|---|---|---|---|
| SP3-01 | Registration | Register with valid details | Account created successfully |
| SP3-02 | Registration | Register with an invalid email | Validation error displayed |
| SP3-03 | Registration | Register using an existing email | Duplicate registration rejected |
| SP3-04 | Login | Login with valid Analyst credentials | Analyst dashboard displayed |
| SP3-05 | Login | Login with valid Admin credentials | Admin dashboard displayed |
| SP3-06 | Login | Login with incorrect password | Error message displayed |
| SP3-07 | Account Security | Login using a suspended account | Login rejected |
| SP3-08 | Role Management | Analyst tries to access Admin functionality | Access denied |
| SP3-09 | Role Management | Admin changes an Analyst into an Admin | Role updated successfully |
| SP3-10 | Account Management | Admin suspends an account | Account suspended successfully |
| SP3-11 | Account Management | Admin reactivates an account | Account activated successfully |
| SP3-12 | Account Security | Admin tries to suspend their own account | Operation rejected |
| SP3-13 | Audit Log | Admin changes another user's role | Action recorded in audit log |
| SP3-14 | Data Privacy | Analyst attempts to access another user's data | Access denied |
| SP3-15 | Model Upload | Upload a valid machine learning model | Model uploaded successfully |
| SP3-16 | Dataset Upload | Upload a valid CSV dataset | Dataset uploaded successfully |
| SP3-17 | Dataset Validation | Upload a dataset with missing target values | Dataset marked invalid |
| SP3-18 | Model Evaluation | Evaluate a compatible classification model | Classification metrics displayed |
| SP3-19 | Model Evaluation | Evaluate a compatible regression model | Regression metrics displayed |
| SP3-20 | Model Evaluation | Select an incompatible task type | Appropriate error displayed |
| SP3-21 | Evaluation History | Open a previously saved evaluation | Previous results displayed |
| SP3-22 | Evaluation Comparison | Compare evaluations of the same task | Comparison results displayed |
| SP3-23 | Classification Charts | View classification evaluation | Confusion matrix and ROC curve displayed |
| SP3-24 | Regression Charts | View regression evaluation | Residual and predicted-vs-actual charts displayed |
| SP3-25 | PDF Reports | Download an evaluation report | PDF report generated successfully |
| SP3-26 | CSRF Security | Submit a request without a valid CSRF token | Request rejected |
| SP3-27 | Error Handling | Submit invalid evaluation details | Appropriate error message displayed |
| SP3-28 | Performance | Evaluate a classification dataset containing 50,000 rows | Evaluation completes within the defined time limit |
| SP3-29 | Performance | Evaluate a regression dataset containing 50,000 rows | Evaluation completes within the defined time limit |

---

## 4. Test Case Execution

The existing automated test suite was executed using Python and pytest.

**Command used:**

```bash
python -m pytest tests -v
```

### Test Execution Results

| Description | Result |
|---|---:|
| Total tests executed | 219 |
| Tests passed | 219 |
| Tests failed | 0 |
| Execution time | 22.67 seconds |
| Overall status | PASSED |

All 219 automated tests passed successfully.

---

## 5. Test Files Used

The following automated test files were reviewed and executed:

| Test File | Purpose |
|---|---|
| test_auth.py | Registration, login, sessions and CSRF |
| test_admin.py | Admin operations and role management |
| test_isolation.py | User data privacy and isolation |
| test_model_service.py | Model upload and validation |
| test_dataset_service.py | Dataset upload and validation |
| test_compatibility_at_upload.py | Model and dataset compatibility |
| test_evaluation_service.py | Model evaluation and metrics |
| test_charts_compare_reports.py | Charts, comparison and PDF reports |
| test_api_workflow.py | Complete application workflow |
| test_performance.py | Performance testing |
| test_app.py | Application page and route testing |

---

## 6. Testing Approach

The following testing approaches are used:

**Unit Testing:** Individual functions and services are tested independently.

**Integration Testing:** Different modules are tested together to verify that they work correctly.

**System Testing:** Complete application workflows are checked.

**Performance Testing:** The application is tested with large datasets to verify execution time.

**Security Testing:** Authentication, authorization, CSRF protection and user data isolation are verified.

---

## 7. Expected Outcomes

- All major application functionalities should work correctly.
- Invalid inputs should produce appropriate error messages.
- Users should only access their own data.
- Admin operations should follow role-based permissions.
- Evaluation results should be calculated and displayed correctly.
- Reports should be generated successfully.
- Large datasets should be processed within the defined performance limit.

---

## 8. Conclusion

The test cases have been updated to cover the latest project functionalities and Sprint changes.

The automated test suite was executed successfully, with 219 tests passed and zero failures.

These test cases help verify the correctness, reliability, security and performance of the Web-Based ML Model Evaluator.

**BL-12 testing and documentation update: Completed.
