
// Comment From Here - Test 1

#define USER_ADDR_SPACE_C_HEADER_FILE  // TODO disable using the other file until tag is updated and https://github.com/efabless/caravel_mgmt_soc_litex/pull/137 is merged

#include <firmware_apis.h>
#include <custom_user_space.h>
#include <ram_info.h>
#include <stdint.h>

// Wait for 'cycles' CPU clock cycles using RISC-V cycle CSR
static inline void wait_cycles(uint32_t cycles)
{
    // Volatile asm prevents the loop from being optimized away.
    for (uint32_t i = 0; i < cycles; i++) {
        __asm__ volatile ("nop");
    }
}

uint32_t read_wishbone(uint32_t);

void main(){
    // Enable management GPIOs as output to use as indicators for finishing configuration  
    ManagmentGpio_outputEnable();
    ManagmentGpio_write(0);
    GPIOs_configureAll(GPIO_MODE_USER_STD_OUT_MONITORED);
    GPIOs_loadConfigs(); // load the configuration 
    User_enableIF(1); // this necessary when reading or writing between wishbone and user project if interface isn't enabled no ack would be recieve and the command will be stuck
    ManagmentGpio_write(1);
    
    volatile int shifting;
    volatile int data_used;
    int start_address[3] = {0, (RAM_NUM_WORDS*4 /10), (RAM_NUM_WORDS*9 /10)};
    int end_address[3] = {(RAM_NUM_WORDS /10), (RAM_NUM_WORDS*5 /10), RAM_NUM_WORDS};
    
    // ---- Single Write + Single Read demo ----
    volatile uint32_t addr = 0x3000000C;
    volatile uint32_t wdata = 0xC21000FF; // [29:25] - Row Address, [24:20] - Column Address, [7:0] Data
    volatile uint32_t wdata1 = 0x42100000;
    volatile uint32_t wdata2 = 0xCA400000;
    volatile uint32_t wdata3 = 0x4A400000;
    
    // Performing Write Operation
    *((volatile uint32_t *) addr) = wdata;
    *((volatile uint32_t *) addr) = wdata1;
    
    wait_cycles(300); // Delay
    
    // Performing Read Operation
    uint32_t temp = read_wishbone(addr);
    
    *((volatile uint32_t *) addr) = wdata2;
    *((volatile uint32_t *) addr) = wdata3;
    *((volatile uint32_t *) addr) = wdata1;
    
    wait_cycles(900); // Delay
    
    uint32_t temp1 = read_wishbone(addr);
    uint32_t temp2 = read_wishbone(addr);
    
    ManagmentGpio_write(0);
}

static unsigned int lfsr = 0xACE1u;  // seed value

int rand(void) {
    // Simple LFSR-based RNG (XOR shift)
    lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
    return (int)(lfsr & 0xFF);  // Return 8-bit random number
}

uint32_t read_wishbone(uint32_t address)
{
    return *(volatile uint32_t *)address;
}

// Comment Till Here - Test 1

// *******************************************************************************************

// // Comment From Here - Test 2

// #define USER_ADDR_SPACE_C_HEADER_FILE  // TODO disable using the other file until tag is updated and https://github.com/efabless/caravel_mgmt_soc_litex/pull/137 is merged

// #include <firmware_apis.h>
// #include <custom_user_space.h>
// #include <ram_info.h>
// #include <stdint.h>

// // Wait for 'cycles' CPU clock cycles using RISC-V cycle CSR
// static inline void wait_cycles(uint32_t cycles)
// {
    // // Volatile asm prevents the loop from being optimized away.
    // for (uint32_t i = 0; i < cycles; i++) {
        // __asm__ volatile ("nop");
    // }
// }

// static inline uint32_t pack_cmd(uint32_t mode, uint32_t row, uint32_t col, uint32_t data20)
// {
    // return ((mode & 0x3) << 30)
         // | ((row  & 0x1F) << 25)
         // | ((col  & 0x1F) << 20)
         // |  (data20 & 0xFFFFF);
// }

// uint32_t read_wishbone(uint32_t);

// void main(){
    // // Enable management GPIOs as output to use as indicators for finishing configuration  
    // ManagmentGpio_outputEnable();
    // ManagmentGpio_write(0);
    // GPIOs_configureAll(GPIO_MODE_USER_STD_OUT_MONITORED);
    // GPIOs_loadConfigs(); // load the configuration 
    // User_enableIF(1); // this necessary when reading or writing between wishbone and user project if interface isn't enabled no ack would be recieve and the command will be stuck
    // ManagmentGpio_write(1);
    
    // volatile int shifting;
    // volatile int data_used;
    // int start_address[3] = {0, (RAM_NUM_WORDS*4 /10), (RAM_NUM_WORDS*9 /10)};
    // int end_address[3] = {(RAM_NUM_WORDS /10), (RAM_NUM_WORDS*5 /10), RAM_NUM_WORDS};
    
    // // ---- Single Write + Single Read demo ----
    // volatile uint32_t addr = 0x3000000C;
    
    // // Performing Write Mode Operation
    // for (uint32_t i = 0; i < 10; i++) {
    // uint32_t data20 = ( (i & 1) == 0 ) ? 0x000FFu : 0x00000u; // alt 0xFF/0x00 in [7:0]
    // uint32_t row    = (i & 0x1F);                              // 5-bit wrap (i=32 -> 0)
    // uint32_t col    = (i & 0x1F);                              // 5-bit wrap
    // uint32_t cmd    = pack_cmd(0x3u, row, col, data20);        // mode=2'b11 (PROGRAM)

    // *(volatile uint32_t *)addr = cmd;

    // }
    
    // wait_cycles(3000); // Delay
    
    // // Performing Read Mode Operation
    // for (uint32_t i = 0; i < 10; i++) {
    // uint32_t data20 = ( (i & 1) == 0 ) ? 0x000FFu : 0x00000u; // alt 0xFF/0x00 in [7:0]
    // uint32_t row    = (i & 0x1F);                              // 5-bit wrap (i=32 -> 0)
    // uint32_t col    = (i & 0x1F);                              // 5-bit wrap
    // uint32_t cmd    = pack_cmd(0x1u, row, col, data20);        // mode=2'b01 (READ)

    // *(volatile uint32_t *)addr = cmd;

    // }
    
    // wait_cycles(500); // Delay
    
    // // Performing Read Operation
    // uint32_t temp[10];
    // for (uint32_t i = 0; i < 10; i++) {
     // temp[i] = read_wishbone(addr);
    // }
    
    // ManagmentGpio_write(0);
// }

// static unsigned int lfsr = 0xACE1u;  // seed value

// int rand(void) {
    // // Simple LFSR-based RNG (XOR shift)
    // lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
    // return (int)(lfsr & 0xFF);  // Return 8-bit random number
// }

// uint32_t read_wishbone(uint32_t address)
// {
    // return *(volatile uint32_t *)address;
// }

// // Comment Till Here - Test 2

// *******************************************************************************************