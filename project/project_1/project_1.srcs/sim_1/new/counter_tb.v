`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2018/08/29 11:32:04
// Design Name: 
// Module Name: counter_tb
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


module counter_tb;
  reg clk;
  reg reset;
  reg enable;
  reg [3:0] count_compare;
  wire [3:0] count;

counter dut(
  .clk(clk),
  .reset(reset),
  .enable(enable),
  .count(count)
);

initial 
  begin
    clk = 0;
    reset = 0;
    enable = 0;
  end

always
  #5 clk = !clk;

initial begin
  $dumpfile("counter.vcd");
  $dumpvars;
end

initial begin
  $display("\t time,\t clk,\t reset,\t enable,\t count");
  $monitor("%d,\t %b,\t %b,\t %b,\t %d",$time,clk,reset,enable,count);
end

initial 
  #300 $finish;

initial begin 
  #10 reset = 1;
  #10 reset = 0;
      enable = 1;
  #130 enable = 0;
  #10 reset = 1;
  #10 reset = 0;
      enable = 1;
  #40 enable = 0;
  #20 enable = 1;
end

always@(posedge clk) begin
  if (reset == 1'b1)
    count_compare <= 0;
  else if (enable == 1'b1)
    count_compare <= count_compare + 1'b1;
end
always@(posedge clk)
  if(count_compare != count) begin
    $display("DUT Error at time %d",$time);
    $display("Expected value %d,Actual value %d",count_compare,count);
   // #5 -> terminate_sim;
  end
endmodule
