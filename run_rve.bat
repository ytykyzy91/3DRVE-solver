@echo off
setlocal EnableExtensions

REM RVE-VAM Windows launcher.
REM Usage examples:
REM   run_rve.bat summary
REM   run_rve.bat test
REM   run_rve.bat run
REM   run_rve.bat large
REM   run_rve.bat fields
REM   run_rve.bat fields-large

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "MESH=%CD%\example\case0001_fix.vtu"
if not exist "%MESH%" set "MESH=%CD%\example\case0001_fix0.vtu"
set "MATERIALS=%CD%\example\user_RVE_analysis.json"
set "OUTPUT=%CD%\outputs\case0001"

if "%~1"=="" goto :help
if /I "%~1"=="help" goto :help
if /I "%~1"=="summary" goto :summary
if /I "%~1"=="test" goto :test
if /I "%~1"=="run" goto :run
if /I "%~1"=="large" goto :large
if /I "%~1"=="fields" goto :fields
if /I "%~1"=="fields-large" goto :fields_large
if /I "%~1"=="fields-homogenization" goto :fields_homogenization

echo Unknown command: %~1
goto :help

:summary
python -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --summary-only
goto :end

:test
python -m compileall -q src tests
if errorlevel 1 goto :end
python -c "import numpy as np; from pathlib import Path; from rve_vam.materials import isotropic_stiffness,load_json,build_phase_mapping,IsotropicMaterial,PhaseMapping; from rve_vam.elements.hex8 import stiffness_hex8,element_average_strain_stress_mises_hex8; from rve_vam.mesh import Mesh; from rve_vam.homogenization import homogenize_mesh; C=isotropic_stiffness(10.0,0.25); cfg=load_json(Path('example/user_RVE_analysis.json')); mp=build_phase_mapping(cfg,np.array([1,2])); pts=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float); Ke,v=stiffness_hex8(pts,isotropic_stiffness(1.0,0.3)); disp=(pts@np.diag([0.01,0,0]).T).reshape(-1); eps,sig,vm,vol=element_average_strain_stress_mises_hex8(pts,disp,C); assert np.allclose(eps,[0.01,0,0,0,0,0]); mesh=Mesh(pts,np.array([[0,1,2,3,4,5,6,7]],dtype=np.int64),'hexahedron',np.array([1]),np.vstack([pts.min(axis=0),pts.max(axis=0)]),1.0); mat=IsotropicMaterial('mat',10.0,0.25,C); res=homogenize_mesh(mesh,PhaseMapping({1:'matrix'},{'matrix':'mat'},{1:mat}),assembly_chunk_size=1,solver='spsolve'); assert np.allclose(res.stiffness,C,rtol=1e-10,atol=1e-10); print('manual verification passed')"
goto :end

:run
python -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 20000 --stiffness-cache-size 4096
goto :end

:large
REM Recommended mode for 1,000,000-scale hexahedral meshes.
REM Increase --assembly-chunk-size if RAM is sufficient; reduce it if memory pressure is high.
python -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 50000 --stiffness-cache-size 20000
goto :end

:fields
REM Use Defs.Composite.LocalFieldsRecovery.MacroStrain from MATERIALS and write VTU field results.
python -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 20000 --stiffness-cache-size 4096 --write-fields --load-steps 1 --field-output-dir "%OUTPUT%\fields" --field-prefix macro_strain
goto :end

:fields_large
REM Large-mesh local-field output. Use --field-output-every to limit VTU file count for many steps.
python -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 50000 --stiffness-cache-size 20000 --write-fields --load-steps 1 --field-output-dir "%OUTPUT%\fields" --field-prefix macro_strain
goto :end

:fields_homogenization
REM Write VTU fields and also run the 6-direction homogenization for stiffness.json/csv.
python -m rve_vam.cli --mesh "%MESH%" --materials "%MATERIALS%" --output "%OUTPUT%" --solver cg --assembly-mode reduced --assembly-chunk-size 20000 --stiffness-cache-size 4096 --write-fields --fields-with-homogenization --load-steps 1 --field-output-dir "%OUTPUT%\fields" --field-prefix macro_strain
goto :end

:help
echo RVE-VAM launcher
echo.
echo Commands:
echo   run_rve.bat summary      Read VTU/JSON and print mesh/material mapping only
echo   run_rve.bat test         Compile code and run built-in small verification
echo   run_rve.bat run          Run stiffness homogenization with reduced sparse assembly
echo   run_rve.bat large        Run stiffness homogenization with 1,000,000-scale settings
echo   run_rve.bat fields       Run macro-strain local-field VTU output only
echo   run_rve.bat fields-large Run macro-strain local-field VTU output only with large-mesh settings
echo   run_rve.bat fields-homogenization Run fields plus 6-direction homogenization stiffness outputs
echo.
echo Edit MESH, MATERIALS, OUTPUT near the top of this BAT for custom files.

:end
if errorlevel 1 (
  echo.
  echo Command failed with exit code %ERRORLEVEL%.
) else (
  echo.
  echo Done.
)
endlocal
