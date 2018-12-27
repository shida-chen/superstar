#!/bin/bash -f
xv_path="/opt/Xilinx/Vivado/Vivado/2016.4"
ExecStep()
{
"$@"
RETVAL=$?
if [ $RETVAL -ne 0 ]
then
exit $RETVAL
fi
}
ExecStep $xv_path/bin/xelab -wto 947aa08d535f4ace9389d2f16872bc26 -m64 --debug typical --relax --mt 8 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip --snapshot counter_tb_behav xil_defaultlib.counter_tb xil_defaultlib.glbl -log elaborate.log
