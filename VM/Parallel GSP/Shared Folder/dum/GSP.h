/* -----------------------------------------------------------------------------
 * GSP API header file
 * for GSP API version 11.4.8.0
 *
 * GSP version 11.2.2.1 - November 2012
 * Important notice: to enable this file to be used by GSPAPI_GenWrapper,
 * obey the following syntax:
 *   - Declare each API function using the GSPAPI_FUNC() macro, which has three
 *     parameters:
 *			name: the name of the API function;
 *			type: the return type;
 *			params: '('-enclosed and ','-separated list of parameter
 *				declaration, including type and parameter name.
 *	 - Use the GSPAPI_FUNC macro on a single line of text.
 *   - Use "#ifdef TEST_VERSION" - "#endif" enclosed blocks to
 *	   indicate optional functions; use no variations!
 *  This file already contains numerous examples.
 *  For further information on the syntax and wrapper generation, contact
 *  Erik.Baalbergen@nlr.nl.
 *
 * 11.3.4.0 - 16 Dec 2013 - Oscar.Kogenhop@nlr.nl
 * 11.3.4.1 - 10 Feb 2014 - Oscar.Kogenhop@nlr.nl
 * 11.3.4.2 - 18 Mar 2014 - Oscar.Kogenhop@nlr.nl
 * 11.4.0.0 - 25 Mar 2014 - Oscar.Kogenhop@nlr.nl
 * 11.4.5.4 - 10 Sep 2015 - Oscar.Kogenhop@nlr.nl
			- Additional functions:
			  + CalculateSteadyStatePoint
			  + InitializeModel
			  + ResetODinputtoDP
			  + GetInputDataListSize
			  + GetOutputDataListSize
			  + GetInputDataList
			  + GetOutputDataList
 * -----------------------------------------------------------------------------
 */

#define GSPAPI_FUNC(name, rtype, params) rtype __cdecl name params // Macro to create the actual function declaration
#define MATLAB_COMMAND_LINE	// effectuate for use in Matlab

#ifndef GSPAPI_DLL_H
#define GSPAPI_DLL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BUILDING_GSPAPI_DLL
#define GSPAPI_DLL __declspec(dllexport)
#else
#define GSPAPI_DLL __declspec(dllimport)
#endif

// -----------------------------------------------------------------------------
// FreeAll
// -----------------------------------------------------------------------------
// Free the loaded forms (get loaded when a dll file is loaded into memory),
// this has to be done prior to unloading a dll.
GSPAPI_FUNC(FreeAll, bool, ());

// -----------------------------------------------------------------------------
// About
// -----------------------------------------------------------------------------
// Displays the About box to show the GSP Version information
GSPAPI_FUNC(About, bool, ());

// -----------------------------------------------------------------------------
// CloseModel
// -----------------------------------------------------------------------------
// Closes the open/active model
// This function has a NoSaveDlg parameter that, if omitted, (default = false)
// or false closes the model without showing the save dialogs When true, and
// if model changes are detected, the save dialog is shown.
//GSPAPI_FUNC(CloseModel, bool, (bool NoSaveDlg));
GSPAPI_FUNC(CloseModel, bool, (bool NoSaveDlg));

// -----------------------------------------------------------------------------
// ModelLoaded
// -----------------------------------------------------------------------------
// Check if a model is loaded
// If ShowMessage : always a message reporting loaded or not is generated
// IF ShowNotLoadedError (active only if ShowMessage=false) :
// if not loaded: error message is generated
GSPAPI_FUNC(ModelLoaded, bool, (bool ShowMessage, bool ShowNotLoadedError));

// -----------------------------------------------------------------------------
// LoadModel
// -----------------------------------------------------------------------------
// Load model opens a model from the model file path and name in the
// ModelFileName character pointer.
// - ModelFileName is a UniCode string type (WCHAR in C and String in Delphi)
// - The ShowModel boolean determines whether the model is shown during loading
#ifndef MATLAB_COMMAND_LINE
GSPAPI_FUNC(LoadModel, bool, (WCHAR *ModelFileName, bool ShowModel));
#endif

// -----------------------------------------------------------------------------
// LoadModelAnsi
// -----------------------------------------------------------------------------
// Identical to LoadModel except that this function only accepts an AnsiString
// ModelFileName.
GSPAPI_FUNC(LoadModelAnsi, bool, (char *ModelFileName, bool ShowModel));

// -----------------------------------------------------------------------------
// RunModel
// -----------------------------------------------------------------------------
// Run the case model
// - StartTimeDlg, to show or hide the start time dialog
// - StabilizeDlg, to show or hide the stabilize dialog
// - Stabilize, to stabilize the simulation before the actual input simulation
// - ShowProgrBox, to show or hide the progress bar window
// This runs the model using the case defined run mode!
GSPAPI_FUNC(RunModel, bool, (bool StartTimeDlg, bool StabilizeDlg, bool Stabilize, bool ShowProgrBox));

// -----------------------------------------------------------------------------
// SaveModel
// -----------------------------------------------------------------------------
// Save the model
// - If SaveAsDlg is true a dialog pops up to request a filename, else
//   the model is saved instantly without dialog popping up.
GSPAPI_FUNC(SaveModel, bool, (bool SaveAsDlg));

// -----------------------------------------------------------------------------
// ShowHideModel
// -----------------------------------------------------------------------------
// For Showing and Hiding GSP model window:
GSPAPI_FUNC(ShowHideModel, bool, (bool ShowModel));

// -----------------------------------------------------------------------------
// EditTransControl
// -----------------------------------------------------------------------------
// Edit the Model's Transient/St.St.series control parameters
GSPAPI_FUNC(EditTransControl, bool, ());

// -----------------------------------------------------------------------------
// EditIterControl
// -----------------------------------------------------------------------------
// Edit the Model's Iteration control parameters
GSPAPI_FUNC(EditIterControl, bool, ());

// -----------------------------------------------------------------------------
// SetTransientTimes
// -----------------------------------------------------------------------------
// SetTransientTimes
//           Starttime: specify to reset the current Transtime value,
//           which is the initial transient/steady-state series
//           time value (i.e. the starting time for the transient)
// 	     MaxEndTime: only used if >0: it then replaces the model's
//           Transient/Series Options setting, which is saved with model
//           settings ! if <0 then Max. End time is taken from existing
//           Transient/Series Options 'Maximum time' setting (std. = 1E20)
GSPAPI_FUNC(SetTransientTimes, bool, (double StartTime, double MaxEndTime));

// -----------------------------------------------------------------------------
// SetInputControlParameterByIndex
// -----------------------------------------------------------------------------
// Use this function to directly SET the value of the specified input variable
// defined in the API interface component.
// INPUT
// - The parameter "ControlVarIndex" specifies the index of the input control
//   parameter defined (the first control parameter has index 1). Note that the
//   index corresponds to the row number (the column header/title is row 0).
// - The "ControlValue" is the actual input parameter value.
GSPAPI_FUNC(SetInputControlParameterByIndex, bool, (int ControlVarIndex, double ControlValue));

// -----------------------------------------------------------------------------
// GetInputControlParameterByIndex
// -----------------------------------------------------------------------------
// Use this function to directly GET the value of the specified input variable
// defined in the API interface component.
// INPUT
// - The parameter "ParamIndex" specifies the index of the input control
//   parameter defined (the first control parameter has index 1). Note that the
//   index corresponds to the row number (the column header/title is row 0).
// OUTPUT
// - The name of the input parameter will be returned in parameter "ParamName".
//   This string includes the component name: e.g. "ManualFuelControl.Wf".
// - The "ParamValue" is the actual value of the input parameter specified in
//   the input parameter grid grid.
GSPAPI_FUNC(GetInputControlParameterByIndex, bool, (int ParamIndex, char *ParamName, double *ParamValue));

// -----------------------------------------------------------------------------
// InitTransient
// -----------------------------------------------------------------------------
// This function will initialize the transient calculations, this function must
// be called once prior to the start of the beginning of the series of transient
// calculations. Basically, this procedure sets and initializes various
// parameters to be able to do transient analyses (set initial start time, end
// time, time step, output interval, various counters, etc.).
// The function takes the following arguments:
// - StabilizeDlg;
//   Controls whether or not the stabilize dialog is shown to the modeler.
// - Stabilize;
//	 Indicates whether the simulation is required to stabilize to the current
//	 input parameters (without the dialog).
// - ShowProgrBox;
//	 Controls whether or not to display the progress box. This should be false
//   as the end time is not known before hand when controlled from other
//   sofware.
// - TransStepStartTime;
//   Initial start time.
// - TimeStep;
//	 Initial value for the time step (dt). Note that for variable solvers, the
//   time step can be overwritten at a later stage.
GSPAPI_FUNC(InitTransient, bool, (bool StabilizeDlg, bool Stabilize, bool ShowProgrBox, double TransStepStartTime, double TimeStep));

// -----------------------------------------------------------------------------
// RunTransientStep
// -----------------------------------------------------------------------------
// This function will calculate a performance point for a specified time using a
// specified time step. Note that the time step may differ from the time step
// defined in InitTransient.
// The function takes the following arguments:
// - ShowProgrBox;
//	 Controls whether or not to display the progress box. This should be false
//   as the end time is not known before hand when controlled from other
//   software.
// - TransStepStartTime;
//   Initial start time.
// - TimeStep;
//	 Initial value for the time step (dt). Note that for variable solvers, the
//   time step can be overwritten at a later stage.// This function will calculate a single transient step.
GSPAPI_FUNC(RunTransientStep, bool, (bool ShowProgrBox, double TransStepStartTime, double TimeStep));

// -----------------------------------------------------------------------------
// CalculateDesignPoint
// -----------------------------------------------------------------------------
// This function will perform a design calculation. This is required for GSP to
// size the engine and calculate relevant design parameters from the user input.
// The function takes the following arguments:
// - Do_Output;
//   If true, output will be generated in the steady state output table.
// - AlwaysErrorOutput;
//   When true, it produces output results when an error has occurred. Note that
//   this can be erroneous non converged data!
// - NoConfirmOutput;
//   If true, output will be written to the table without showing a message
//   dialog.
GSPAPI_FUNC(CalculateDesignPoint, bool, (bool Do_Output, bool AlwaysErrorOutput, bool NoConfirmOutput));


// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// CalculateSteadyStatePoint
// -----------------------------------------------------------------------------
// This function will perform a steady state calculation. The user requires to
// set the steady state input data prior to running this function using
// functions SetInputControlArray or preferred: SetInputControlParameterByIndex.
// The function takes the following arguments:
// - DoShowProgBox;
//   If true, the iteration progress box will be shown on the screen.
// - DoTableOutput;
//   If true, output will be generated in the steady state output table.
GSPAPI_FUNC(CalculateSteadyStatePoint, bool, (bool DoShowProgBox, bool DoTableOutput));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// InitializeModel
// -----------------------------------------------------------------------------
// This function will initialize the model
// The function has no result tyoe and does not take arguments.
GSPAPI_FUNC(InitializeModel, void, ());

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// ResetODinputtoDP
// -----------------------------------------------------------------------------
// Function to Reset OD input to DP model input
// The function has no result type and has one argument:
// - DoConfirm;
//   If set to false, no confirmation will be asked whether the user wants to
//   override the current input data as it is about to be overwritten by the
//   design input data.
GSPAPI_FUNC(ResetODinputtoDP, void, (bool DoConfirm));

// -----------------------------------------------------------------------------
// GetInputControlParameterArraySize
// -----------------------------------------------------------------------------
// GetInputControlParameterArraySize retrieves the amount of the input control
// parameters defined in the API interface component.
// The function will return the amount of parameters in parameter
// InputArraySize.
GSPAPI_FUNC(GetInputControlParameterArraySize, bool, (int *InputArraySize));


// -----------------------------------------------------------------------------
// GetOutputDataParameterArraySize
// -----------------------------------------------------------------------------
// GetOutputDataParameterArraySize retrieves the amount of the output control
// parameters defined in the grid inside the API interface component
// The function will return the amount of parameters in parameter
// OutputArraySize.
GSPAPI_FUNC(GetOutputDataParameterArraySize, bool, (int *OutputArraySize));

// -----------------------------------------------------------------------------
// SetInputControlArray
// -----------------------------------------------------------------------------
// SetInputControlArray sets an array of input parameter values in the input
// parameter grid where the array element will be stored at the corresponding
// row (array element 1 will be stored at row 1, etc.).
// The input is a Delphi array. Note that Delphi arrays are incompatible with
// array definitions in C (only referring to the first element). Mapping Delphi
// arrays in C(++)-code requires the use of extra coding, by mimicing the Delphi
// memory storage in C-code. Therefore the function below cannot be called
// directly! If so, the dll will crash!
//
// NOTICE: Usage of this function is strongly disencouraged without proper
// understanding of Delphi and C memory management.
//
// The function argument is a pointer to a character, which points to a certain
// location in the memory that has been designed to store an array in Delphi
// style.
// Instead of using this function it is advised to code a loop using function
// SetInputControlParameter.
GSPAPI_FUNC(SetInputControlArray, bool, (char *InputValArray));

// -----------------------------------------------------------------------------
// GetOutputDataParameterValueByIndex
// -----------------------------------------------------------------------------
// GetOutputDataParameterValueByIndex gets the value of the specified output
// parameter defined in the API interface component by the index in the output
// data grid.
// INPUT
// - The parameter "ParamIndex" specifies the index of the output parameter
//   defined in the output grid(the first control parameter has index 1).
//   Note that the index corresponds to the row number (the column header/title
//	 is row 0).
// OUTPUT
// - The value of the parameter at index "ParamIndex" will be stored in the
//   variable "ParamValue". The grid stores the last results of a finished and
//   valid calculation.
GSPAPI_FUNC(GetOutputDataParameterValueByIndex, bool, (int ParamIndex, double *ParamValue, bool DoScale));

// -----------------------------------------------------------------------------
// GetOutputDataArray
// -----------------------------------------------------------------------------
// GetOutputDataArray gets an array of output parameter values from the output
// parameter grid where the array element will be obtained from the
// corresponding row (array element 1 will be obtained from row 1, etc.).
// The output is a Delphi array. Note that Delphi arrays are incompatible with
// array definitions in C (only referring to the first element). Mapping Delphi
// arrays in C(++)-code requires the use of extra coding, by mimicking the Delphi
// memory storage in C-code. Therefore the function below cannot be called
// directly! If so, the dll will crash!
//
// NOTICE: Usage of this function is strongly discouraged without proper
// understanding of Delphi and C memory management.
//
// The function argument is a pointer to a character, which points to a certain
// location in the memory that has been designed to store an array in Delphi
// style.
// Instead of using this function it is advised to code a loop using function
// GetOutputDataParameterValueByIndex.
GSPAPI_FUNC(GetOutputDataArray, bool, (char *OutputValArray));

// -----------------------------------------------------------------------------
// ConfigureModel
// -----------------------------------------------------------------------------
// Use this function to automatically configure the model such that the model
// options are configured optimally for use with the API (i.e. no table pop-up
// for output data, reports, etc.
GSPAPI_FUNC(ConfigureModel, bool, ());

// -----------------------------------------------------------------------------
// ProgramInfo
// -----------------------------------------------------------------------------
// This function returns basic program information (to be extended, see ARP4868)
GSPAPI_FUNC(ProgramInfo, bool, (char *Name, char *Version));

// -----------------------------------------------------------------------------
// ShowConvergenceMonitor
// -----------------------------------------------------------------------------
// Function to show and hide the "Convergence monitor". The convergence monitor
// window graphically displays the error convergence during simulations.
GSPAPI_FUNC(ShowConvergenceMonitor, bool, (bool DoShow));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// GetInputDataListSize
// -----------------------------------------------------------------------------
// Function to retrieve the amount of input parameters from the model. The
// amount will be returned in listSize. Note that the length of the longest
// string is returned in maxStringSize (unless maxStringSize equals the null
// pointer).
GSPAPI_FUNC(GetInputDataListSize, bool, (int *listSize, int *maxSizeString));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// GetOutputDataListSize
// -----------------------------------------------------------------------------
// Function to retrieve the amount of output parameters from the model. The
// amount will be returned in listSize. Note that the length of the longest
// string is returned in maxStringSize (unless maxStringSize equals the null
// pointer).
GSPAPI_FUNC(GetOutputDataListSize, bool, (int *listSize, int *maxSizeString));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// GetInputDataList
// -----------------------------------------------------------------------------
// Function to retrieve the amount of input parameters, input values, units and
// the row index from the model.  Note that the char ** need to be correctly
// allocated prior to running this function (e.g. use the GetInputDataListSize
// function to get the maximum length of the largest string in the list, the
// longest possible unit string is about 11, unless custom defined units are
// used)
// OUTPUT:
// - paramList; List of parameter names
// - valueList; List of input values, must all be scalar and of the same type
// - unitList; List of unit strings
// - indexList; List of index nr's in the grid (actually row number)
// INPUT:
// - size; Number of names in the list (from GetInputDataListSize)
GSPAPI_FUNC(GetInputDataList, bool, (char **paramList, double *valueList, char **unitList, int *indexList, int size));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// GetOutputDataList
// -----------------------------------------------------------------------------
// Function to retrieve the amount of output parameters, values, units and the
// row index from the model.  Note that the char ** need to be correctly
// allocated prior to running this function (e.g. use the GetOutputDataListSize
// function to get the maximum length of the largest string in the list, the
// longest possible unit string is about 11, unless custom defined units are
// used)
// OUTPUT:
// - paramList; List of parameter names
// - valueList; List of input values, must all be scalar and of the same type
// - unitList; List of unit strings
// - indexList; List of index nr's in the grid (actually row number)
// INPUT:
// - size; Number of names in the list (from GetOutputDataListSize)
GSPAPI_FUNC(GetOutputDataList, bool, (char **paramList, double *valueList, char **unitList, int *indexList, int size));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// GetStStTableSize
// -----------------------------------------------------------------------------
// This function retrieves the amount of characters that need to be allocated
// to store the StStOutputTable to string.
// OUTPUT:
// - maxLines; Number of lines in the report
// - Length of largest char string
GSPAPI_FUNC(GetStStTableSize, bool, (int *maxLines, int *maxSizeString));

// 11.4.5.4 OK
// -----------------------------------------------------------------------------
// GetStStTable
// -----------------------------------------------------------------------------
// This function stores the steady state table in a predefined (allocated) char
// array that is passed by a pointer to a char pointer.
// INPUT: (from GetStStTableSize)
// - maxLines; Number of lines in the report
// - Length of largest char string
// OUTPUT:
// - StStReport; List of parameter names
GSPAPI_FUNC(GetStStTable, bool, (char **StStReport, int maxLines, int maxSizeString));

#ifdef TEST_VERSION
// API Test Functions
GSPAPI_FUNC(AboutToClose, bool, ());
GSPAPI_FUNC(AboutToCloseIn, bool, (int p1));
GSPAPI_FUNC(InputAndOutputInt, bool, (int p1, int *p2));
GSPAPI_FUNC(InputArrayDouble, bool, (char *p1, double *p2));
GSPAPI_FUNC(InputBool, bool, (bool p1, bool p2));
GSPAPI_FUNC(InputDouble, bool, (double p1, double p2, int p3));
GSPAPI_FUNC(InputInt, bool, (int p1, int p2));
#ifndef MATLAB_COMMAND_LINE
GSPAPI_FUNC(InputString, bool, (WCHAR *p1, WCHAR *p2));
#endif
GSPAPI_FUNC(OutputArrayDouble, bool, (char *p1));
#ifndef MATLAB_COMMAND_LINE
GSPAPI_FUNC(OutputString, bool, (WCHAR *p1, int *p2));
#endif
GSPAPI_FUNC(OutputVar, bool, (int *p1, double *p2, bool *p3, double *p4));
#endif



#ifdef __cplusplus
}
#endif

#endif // !defined(GSPAPI_DLL_H)