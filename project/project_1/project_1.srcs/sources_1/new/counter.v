`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2018/08/29 11:22:26
// Design Name: 
// Module Name: counter
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module counter(
 input enable,
 input clk,
 input reset,
 output [3 : 0] count
 );
 reg [3 : 0] cnt;
always@(posedge clk) begin
if (reset == 1'b1)
    cnt <= 0;
else if (enable == 1'b1) begin
    if (cnt == 12)
        cnt <= 0;
    else 
        cnt <= cnt + 1;
end
end
assign count = cnt;
endmodule
