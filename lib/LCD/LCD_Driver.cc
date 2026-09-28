/*****************************************************************************
* | File      	:	LCD_Driver.c
* | Author      :   Luckfox team
* | Function    :	LCD Drive function
* | Info        :
*   Image scanning
*      Please use progressive scanning to generate images or fonts
*----------------
* |	This version:   V1.0
* | Date        :   2023-09-05
* | Info        :   Basic version
*
******************************************************************************/

/**************************Intermediate driver layer**************************/
#include "LCD_Driver.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LCD_DIS sLCD_DIS;
uint8_t id;
/*******************************************************************************
function:
	Hardware reset
*******************************************************************************/
static void LCD_Reset(void)
{
    DEV_Digital_Write(LCD_RST_PIN,1);   
    DEV_Delay_ms(500);
    DEV_Digital_Write(LCD_RST_PIN,0);
    DEV_Delay_ms(500);
    DEV_Digital_Write(LCD_RST_PIN,1);
    DEV_Delay_ms(500);
}

static void LCD_SetBackLight(uint16_t value)
{
//	PWM_SetValue(value);
}

static void LCD_WriteImageBytes(const uint8_t *data, uint32_t len)
{
	uint32_t line_bytes = 0;
	if (id == LCD_2_8) {
		line_bytes = LCD_2_8_WIDTH * 2u;
	} else if (id == LCD_1_8) {
		line_bytes = LCD_1_8_WIDTH * 2u;
	} else {
		line_bytes = LCD_3_5_WIDTH * 2u;
	}

	static uint32_t cached_chunk_limit = 0;
	if (cached_chunk_limit == 0) {
		uint32_t spidev_bufsiz = 4096u;
		FILE *fp = fopen("/sys/module/spidev/parameters/bufsiz", "r");
		if (fp) {
			unsigned long value = 0;
			if (fscanf(fp, "%lu", &value) == 1 && value >= 512ul) {
				spidev_bufsiz = (uint32_t)value;
			}
			fclose(fp);
		}

		uint32_t preferred = line_bytes ? line_bytes * 64u : spidev_bufsiz;
		if (preferred > spidev_bufsiz) preferred = spidev_bufsiz;
		if (preferred & 1u) {
			preferred--;
		}
		if (preferred < 512u) preferred = 512u;
		cached_chunk_limit = preferred;
		printf("[LCD] SPI write chunk=%u bytes (spidev bufsiz=%u)\n",
		       cached_chunk_limit, spidev_bufsiz);
	}

	const uint32_t chunk_limit = cached_chunk_limit;

	while (len > 0) {
		uint32_t chunk = len > chunk_limit ? chunk_limit : len;
		DEV_SPI_Write_nByte((uint8_t *)data, chunk);
		data += chunk;
		len -= chunk;
	}
}

static void LCD_WriteRegDataBytes(uint8_t reg, const uint8_t *data, uint32_t len)
{
	DEV_Digital_Write(LCD_DC_PIN, 0);
	DEV_Digital_Write(LCD_CS_PIN, 0);
	DEV_SPI_WriteByte(reg);
	DEV_Digital_Write(LCD_CS_PIN, 1);

	if (data && len > 0) {
		DEV_Digital_Write(LCD_DC_PIN, 1);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_SPI_Write_nByte((uint8_t *)data, len);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	}
}

static int lcd_force_2_8_requested(void)
{
    const char *force_2_8 = getenv("ATTENDANCE_LCD_FORCE_2_8");
    if (!force_2_8 || !force_2_8[0]) return 0;
    return !(force_2_8[0] == '0' && force_2_8[1] == '\0');
}

/*******************************************************************************
function:
		Write register address and data
*******************************************************************************/
void LCD_WriteReg(uint8_t Reg)
{
    DEV_Digital_Write(LCD_DC_PIN,0);
	DEV_Digital_Write(LCD_CS_PIN,0);
    DEV_SPI_WriteByte(Reg);
	DEV_Digital_Write(LCD_CS_PIN,1);
}

void LCD_WriteData(uint16_t Data)
{
	if(LCD_2_8 == id || LCD_1_8 == id){
		DEV_Digital_Write(LCD_DC_PIN,1);
		DEV_Digital_Write(LCD_CS_PIN,0);
		DEV_SPI_WriteByte((uint8_t)Data);
		DEV_Digital_Write(LCD_CS_PIN,1);
	}else{
		DEV_Digital_Write(LCD_DC_PIN,1);
		DEV_Digital_Write(LCD_CS_PIN,0);
		DEV_SPI_WriteByte(Data >> 8);
		DEV_SPI_WriteByte(Data & 0XFF);
		DEV_Digital_Write(LCD_CS_PIN,1);
	}
}

/*******************************************************************************
function:
		Common register initialization
*******************************************************************************/
static void LCD_InitReg(void)
{
    if (lcd_force_2_8_requested()) {
        id = LCD_2_8;
        printf("[LCD] force 2.8 SPI ILI9341 init\n");
    } else {
        id = LCD_Read_Id();
        if (id != LCD_1_8) {
            id = LCD_2_8;
        }
    }
	if(LCD_2_8 == id){
		// 2.8" SPI ILI9341 initialization, matched to Demo_STM32F103RCT6_Software_SPI.
		LCD_WriteReg(0xCF);
		LCD_WriteData(0x00);
		LCD_WriteData(0xC9);
		LCD_WriteData(0x30);
		LCD_WriteReg(0xED);
		LCD_WriteData(0x64);
		LCD_WriteData(0x03);
		LCD_WriteData(0x12);
		LCD_WriteData(0x81);
		LCD_WriteReg(0xE8);
		LCD_WriteData(0x85);
		LCD_WriteData(0x10);
		LCD_WriteData(0x7A);
		LCD_WriteReg(0xCB);
		LCD_WriteData(0x39);
		LCD_WriteData(0x2C);
		LCD_WriteData(0x00);
		LCD_WriteData(0x34);
		LCD_WriteData(0x02);
		LCD_WriteReg(0xF7);
		LCD_WriteData(0x20);
		LCD_WriteReg(0xEA);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteReg(0xC0);
		LCD_WriteData(0x1B);
		LCD_WriteReg(0xC1);
		LCD_WriteData(0x00);
		LCD_WriteReg(0xC5);
		LCD_WriteData(0x30);
		LCD_WriteData(0x30);
		LCD_WriteReg(0xC7);
		LCD_WriteData(0xB7);
		LCD_WriteReg(0x36);
		LCD_WriteData(0x08);
		LCD_WriteReg(0x3A);
		LCD_WriteData(0x55);
		LCD_WriteReg(0xB1);
		LCD_WriteData(0x00);
		LCD_WriteData(0x1A);
		LCD_WriteReg(0xB6);
		LCD_WriteData(0x0A);
		LCD_WriteData(0xA2);
		LCD_WriteReg(0xF2);
		LCD_WriteData(0x00);
		LCD_WriteReg(0x26);
		LCD_WriteData(0x01);
		LCD_WriteReg(0xE0);
		LCD_WriteData(0x0F);
		LCD_WriteData(0x2A);
		LCD_WriteData(0x28);
		LCD_WriteData(0x08);
		LCD_WriteData(0x0E);
		LCD_WriteData(0x08);
		LCD_WriteData(0x54);
		LCD_WriteData(0xA9);
		LCD_WriteData(0x43);
		LCD_WriteData(0x0A);
		LCD_WriteData(0x0F);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteReg(0xE1);
		LCD_WriteData(0x00);
		LCD_WriteData(0x15);
		LCD_WriteData(0x17);
		LCD_WriteData(0x07);
		LCD_WriteData(0x11);
		LCD_WriteData(0x06);
		LCD_WriteData(0x2B);
		LCD_WriteData(0x56);
		LCD_WriteData(0x3C);
		LCD_WriteData(0x05);
		LCD_WriteData(0x10);
		LCD_WriteData(0x0F);
		LCD_WriteData(0x3F);
		LCD_WriteData(0x3F);
		LCD_WriteData(0x0F);
		LCD_WriteReg(0x2B);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteData(0x01);
		LCD_WriteData(0x3F);
		LCD_WriteReg(0x2A);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteData(0x00);
		LCD_WriteData(0xEF);
		LCD_WriteReg(0x11);
		DEV_Delay_ms(120);
		LCD_WriteReg(0x29);
	}else if (LCD_1_8 == id) {
		// 1.8" ST7735-like SPI TFT initialization
		LCD_WriteReg(0x01); // Software reset
		DEV_Delay_ms(150);
		LCD_WriteReg(0x11); // Sleep out
		DEV_Delay_ms(120);
		LCD_WriteReg(0xB1);
		LCD_WriteData(0x01);
		LCD_WriteData(0x2C);
		LCD_WriteData(0x2D);
		LCD_WriteReg(0xB2);
		LCD_WriteData(0x01);
		LCD_WriteData(0x2C);
		LCD_WriteData(0x2D);
		LCD_WriteReg(0xB3);
		LCD_WriteData(0x01);
		LCD_WriteData(0x2C);
		LCD_WriteData(0x2D);
		LCD_WriteData(0x01);
		LCD_WriteData(0x2C);
		LCD_WriteData(0x2D);
		LCD_WriteReg(0xB4);
		LCD_WriteData(0x07);
		LCD_WriteReg(0xC0);
		LCD_WriteData(0xA2);
		LCD_WriteData(0x02);
		LCD_WriteData(0x84);
		LCD_WriteReg(0xC1);
		LCD_WriteData(0xC5);
		LCD_WriteReg(0xC2);
		LCD_WriteData(0x0A);
		LCD_WriteData(0x00);
		LCD_WriteReg(0xC3);
		LCD_WriteData(0x8A);
		LCD_WriteData(0x2A);
		LCD_WriteReg(0xC4);
		LCD_WriteData(0x8A);
		LCD_WriteData(0xEE);
		LCD_WriteReg(0xC5);
		LCD_WriteData(0x0E);
		LCD_WriteReg(0x36);
		LCD_WriteData(0xC0);
		LCD_WriteReg(0x3A);
		LCD_WriteData(0x05);
		LCD_WriteReg(0x29);
	}else{
		LCD_WriteReg(0x21);
		LCD_WriteReg(0xC2);	//Normal mode, increase can change the display quality, while increasing power consumption
		LCD_WriteData(0x33);
		LCD_WriteReg(0XC5);
		LCD_WriteData(0x00);
		LCD_WriteData(0x1e);//VCM_REG[7:0]. <=0X80.
		LCD_WriteData(0x80);
		LCD_WriteReg(0xB1);//Sets the frame frequency of full color normal mode
		LCD_WriteData(0xB0);//0XB0 =70HZ, <=0XB0.0xA0=62HZ
		LCD_WriteReg(0x36);
		LCD_WriteData(0x28); //2 DOT FRAME MODE,F<=70HZ.
		LCD_WriteReg(0XE0);
		LCD_WriteData(0x0);
		LCD_WriteData(0x13);
		LCD_WriteData(0x18);
		LCD_WriteData(0x04);
		LCD_WriteData(0x0F);
		LCD_WriteData(0x06);
		LCD_WriteData(0x3a);
		LCD_WriteData(0x56);
		LCD_WriteData(0x4d);
		LCD_WriteData(0x03);
		LCD_WriteData(0x0a);
		LCD_WriteData(0x06);
		LCD_WriteData(0x30);
		LCD_WriteData(0x3e);
		LCD_WriteData(0x0f);		
		LCD_WriteReg(0XE1);
		LCD_WriteData(0x0);
		LCD_WriteData(0x13);
		LCD_WriteData(0x18);
		LCD_WriteData(0x01);
		LCD_WriteData(0x11);
		LCD_WriteData(0x06);
		LCD_WriteData(0x38);
		LCD_WriteData(0x34);
		LCD_WriteData(0x4d);
		LCD_WriteData(0x06);
		LCD_WriteData(0x0d);
		LCD_WriteData(0x0b);
		LCD_WriteData(0x31);
		LCD_WriteData(0x37);
		LCD_WriteData(0x0f);
		LCD_WriteReg(0X3A);	//Set Interface Pixel Format
		LCD_WriteData(0x55);
		LCD_WriteReg(0x11);//sleep out
		DEV_Delay_ms(120);
		LCD_WriteReg(0x29);//Turn on the LCD display
	}
}

/********************************************************************************
function:	Set the display scan and color transfer modes
parameter:
		Scan_dir   :   Scan direction
		Colorchose :   RGB or GBR color format
********************************************************************************/
void LCD_SetGramScanWay(LCD_SCAN_DIR Scan_dir)
{
    uint16_t MemoryAccessReg_Data = 0; //addr:0x36
    uint16_t DisFunReg_Data = 0; //addr:0xB6

	if(LCD_2_8 == id){
		switch(Scan_dir){
			case L2R_U2D: MemoryAccessReg_Data = 0x08; break;
			case D2U_L2R: MemoryAccessReg_Data = 0x68; break;
			case R2L_D2U: MemoryAccessReg_Data = 0xC8; break;
			case U2D_R2L: MemoryAccessReg_Data = 0xA8; break;
		}
		sLCD_DIS.LCD_Scan_Dir = Scan_dir;
		if(Scan_dir == L2R_U2D || Scan_dir == R2L_D2U) {
			sLCD_DIS.LCD_Dis_Column = LCD_2_8_WIDTH;
			sLCD_DIS.LCD_Dis_Page   = LCD_2_8_HEIGHT;
		} else {
			sLCD_DIS.LCD_Dis_Column = LCD_2_8_HEIGHT;
			sLCD_DIS.LCD_Dis_Page   = LCD_2_8_WIDTH;
		}
		LCD_WriteReg(0x36);
		LCD_WriteData(MemoryAccessReg_Data);
	} else if (LCD_1_8 == id) {
		// ST7735S: MX|MY|MV bits in MADCTL (0x36)
		switch(Scan_dir){
			case L2R_U2D: MemoryAccessReg_Data = 0x00; break;
			case D2U_L2R: MemoryAccessReg_Data = 0xA0; break;
			case R2L_D2U: MemoryAccessReg_Data = 0xC0; break;
			case U2D_R2L: MemoryAccessReg_Data = 0x60; break;
		}
		sLCD_DIS.LCD_Scan_Dir = Scan_dir;
		if(Scan_dir == L2R_U2D || Scan_dir == R2L_D2U) {
			sLCD_DIS.LCD_Dis_Column = LCD_1_8_WIDTH;
			sLCD_DIS.LCD_Dis_Page   = LCD_1_8_HEIGHT;
		} else {
			sLCD_DIS.LCD_Dis_Column = LCD_1_8_HEIGHT;
			sLCD_DIS.LCD_Dis_Page   = LCD_1_8_WIDTH;
		}
		LCD_WriteReg(0x36);
		LCD_WriteData(MemoryAccessReg_Data);
	} else {
		//Pico-ResTouch-LCD-3.5
		switch (Scan_dir) {
		case L2R_U2D: MemoryAccessReg_Data = 0x08; DisFunReg_Data = 0x22; break;
		case R2L_D2U: MemoryAccessReg_Data = 0x08; DisFunReg_Data = 0x42; break;
		case U2D_R2L: MemoryAccessReg_Data = 0x28; DisFunReg_Data = 0x02; break;
		case D2U_L2R: MemoryAccessReg_Data = 0x28; DisFunReg_Data = 0x62; break;
		}
		sLCD_DIS.LCD_Scan_Dir = Scan_dir;
		if(Scan_dir == L2R_U2D || Scan_dir == R2L_D2U) {
			sLCD_DIS.LCD_Dis_Column = LCD_3_5_HEIGHT;
			sLCD_DIS.LCD_Dis_Page   = LCD_3_5_WIDTH;
		} else {
			sLCD_DIS.LCD_Dis_Column = LCD_3_5_WIDTH;
			sLCD_DIS.LCD_Dis_Page   = LCD_3_5_HEIGHT;
		}
		LCD_WriteReg(0xB6);
		LCD_WriteData(0X00);
		LCD_WriteData(DisFunReg_Data);
		LCD_WriteReg(0x36);
		LCD_WriteData(MemoryAccessReg_Data);
	}
}

/********************************************************************************
function:	Set the display scan and color transfer modes
parameter:
		Scan_dir   :   Scan direction
		Colorchose :   RGB or GBR color format
********************************************************************************/
void BMP_SetGramScanWay(LCD_SCAN_DIR Scan_dir)
{
    uint16_t MemoryAccessReg_Data = 0; //addr:0x36
    uint16_t DisFunReg_Data = 0; //addr:0xB6

	if(LCD_2_8 == id){
		switch(Scan_dir){
			case L2R_U2D: MemoryAccessReg_Data = 0x40; break;
			case D2U_L2R: MemoryAccessReg_Data = 0x20; break;
			case R2L_D2U: MemoryAccessReg_Data = 0x80; break;
			case U2D_R2L: MemoryAccessReg_Data = 0xE0; break;
		}
		sLCD_DIS.LCD_Scan_Dir = Scan_dir;
		if(Scan_dir == L2R_U2D || Scan_dir == R2L_D2U) {
			sLCD_DIS.LCD_Dis_Column = LCD_2_8_WIDTH;
			sLCD_DIS.LCD_Dis_Page   = LCD_2_8_HEIGHT;
		} else {
			sLCD_DIS.LCD_Dis_Column = LCD_2_8_HEIGHT;
			sLCD_DIS.LCD_Dis_Page   = LCD_2_8_WIDTH;
		}
		LCD_WriteReg(0x36);
		LCD_WriteData(MemoryAccessReg_Data);
	} else if (LCD_1_8 == id) {
		switch(Scan_dir){
			case L2R_U2D: MemoryAccessReg_Data = 0x40; break;
			case D2U_L2R: MemoryAccessReg_Data = 0x20; break;
			case R2L_D2U: MemoryAccessReg_Data = 0x80; break;
			case U2D_R2L: MemoryAccessReg_Data = 0xE0; break;
		}
		sLCD_DIS.LCD_Scan_Dir = Scan_dir;
		if(Scan_dir == L2R_U2D || Scan_dir == R2L_D2U) {
			sLCD_DIS.LCD_Dis_Column = LCD_1_8_WIDTH;
			sLCD_DIS.LCD_Dis_Page   = LCD_1_8_HEIGHT;
		} else {
			sLCD_DIS.LCD_Dis_Column = LCD_1_8_HEIGHT;
			sLCD_DIS.LCD_Dis_Page   = LCD_1_8_WIDTH;
		}
		LCD_WriteReg(0x36);
		LCD_WriteData(MemoryAccessReg_Data);
	} else {
		//Pico-ResTouch-LCD-3.5
		switch (Scan_dir) {
		case L2R_U2D: MemoryAccessReg_Data = 0x48; DisFunReg_Data = 0x22; break;
		case R2L_D2U: MemoryAccessReg_Data = 0x48; DisFunReg_Data = 0x42; break;
		case U2D_R2L: MemoryAccessReg_Data = 0xA8; DisFunReg_Data = 0x02; break;
		case D2U_L2R: MemoryAccessReg_Data = 0xA8; DisFunReg_Data = 0x62; break;
		}
		sLCD_DIS.LCD_Scan_Dir = Scan_dir;
		if(Scan_dir == L2R_U2D || Scan_dir == R2L_D2U) {
			sLCD_DIS.LCD_Dis_Column = LCD_3_5_HEIGHT;
			sLCD_DIS.LCD_Dis_Page   = LCD_3_5_WIDTH;
		} else {
			sLCD_DIS.LCD_Dis_Column = LCD_3_5_WIDTH;
			sLCD_DIS.LCD_Dis_Page   = LCD_3_5_HEIGHT;
		}
		LCD_WriteReg(0xB6);
		LCD_WriteData(0X00);
		LCD_WriteData(DisFunReg_Data);
		LCD_WriteReg(0x36);
		LCD_WriteData(MemoryAccessReg_Data);
	}
}

/********************************************************************************
function:
	initialization
********************************************************************************/
void LCD_Init(LCD_SCAN_DIR LCD_ScanDir, uint16_t LCD_BLval)
{
    
    LCD_Reset();//Hardware reset

    LCD_InitReg();//Set the initialization register
	
	if(LCD_BLval > 1000)
		LCD_BLval = 1000;
	//LCD_SetBackLight(LCD_BLval);
	
	LCD_SetGramScanWay(LCD_ScanDir);//Set the display scan and color transfer modes
	DEV_Delay_ms(200);
}

/********************************************************************************
function:	Sets the start position and size of the display area
parameter:
	Xstart 	:   X direction Start coordinates
	Ystart  :   Y direction Start coordinates
	Xend    :   X direction end coordinates
	Yend    :   Y direction end coordinates
********************************************************************************/
void LCD_SetWindow(POINT Xstart, POINT Ystart,	POINT Xend, POINT Yend)
{
	if (LCD_2_8 == id || LCD_1_8 == id) {
		uint8_t x_data[4] = {
			(uint8_t)(Xstart >> 8),
			(uint8_t)(Xstart & 0xff),
			(uint8_t)((Xend - 1) >> 8),
			(uint8_t)((Xend - 1) & 0xff)
		};
		uint8_t y_data[4] = {
			(uint8_t)(Ystart >> 8),
			(uint8_t)(Ystart & 0xff),
			(uint8_t)((Yend - 1) >> 8),
			(uint8_t)((Yend - 1) & 0xff)
		};
		LCD_WriteRegDataBytes(0x2A, x_data, sizeof(x_data));
		LCD_WriteRegDataBytes(0x2B, y_data, sizeof(y_data));
		LCD_WriteRegDataBytes(0x2C, nullptr, 0);
		return;
	}

	//set the X coordinates
	LCD_WriteReg(0x2A);
	LCD_WriteData(Xstart >> 8);	 		//Set the horizontal starting point to the high octet
	LCD_WriteData(Xstart & 0xff);	 	//Set the horizontal starting point to the low octet
	LCD_WriteData((Xend - 1) >> 8);		//Set the horizontal end to the high octet
	LCD_WriteData((Xend - 1) & 0xff);	//Set the horizontal end to the low octet

	//set the Y coordinates
	LCD_WriteReg(0x2B);
	LCD_WriteData(Ystart >> 8);
	LCD_WriteData(Ystart & 0xff );
	LCD_WriteData((Yend - 1) >> 8);
	LCD_WriteData((Yend - 1) & 0xff);

    LCD_WriteReg(0x2C);
}

/********************************************************************************
function:	Set the display point (Xpoint, Ypoint)
parameter:
	xStart :   X direction Start coordinates
	xEnd   :   X direction end coordinates
********************************************************************************/
void LCD_SetCursor(POINT Xpoint, POINT Ypoint)
{
	LCD_SetWindow(Xpoint, Ypoint, Xpoint, Ypoint);
}

/******************************************************************************
function:	Clear screen function, refresh the screen to a certain color
parameter	:
	  Color :		The color you want to clear all the screen
******************************************************************************/
void LCD_Clear(UWORD Color)
{
	unsigned int j;

	if(LCD_2_8==id){
		UWORD Image[LCD_2_8_WIDTH*LCD_2_8_HEIGHT];
		Color = ((Color<<8)&0xff00)|(Color>>8);
		for (j = 0; j < LCD_2_8_HEIGHT*LCD_2_8_WIDTH; j++) Image[j] = Color;
		LCD_SetWindow(0,0,LCD_2_8_WIDTH,LCD_2_8_HEIGHT);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_Digital_Write(LCD_DC_PIN, 1);
		LCD_WriteImageBytes((uint8_t *)Image, (uint32_t)LCD_2_8_WIDTH * LCD_2_8_HEIGHT * 2u);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	} else if (LCD_1_8==id) {
		UWORD Image[LCD_1_8_WIDTH*LCD_1_8_HEIGHT];
		Color = ((Color<<8)&0xff00)|(Color>>8);
		for (j = 0; j < LCD_1_8_HEIGHT*LCD_1_8_WIDTH; j++) Image[j] = Color;
		LCD_SetWindow(0,0,LCD_1_8_WIDTH,LCD_1_8_HEIGHT);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_Digital_Write(LCD_DC_PIN, 1);
		LCD_WriteImageBytes((uint8_t *)Image, (uint32_t)LCD_1_8_WIDTH * LCD_1_8_HEIGHT * 2u);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	} else {
		UWORD Image[LCD_3_5_WIDTH*LCD_3_5_HEIGHT];
		Color = ((Color<<8)&0xff00)|(Color>>8);
		for (j = 0; j < LCD_3_5_HEIGHT*LCD_3_5_WIDTH; j++) Image[j] = Color;
		LCD_SetWindow(0,0,0,0);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_Digital_Write(LCD_DC_PIN, 1);
		LCD_WriteImageBytes((uint8_t *)Image, (uint32_t)LCD_3_5_WIDTH * LCD_3_5_HEIGHT * 2u);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	}
}

void Handler_LCD(int signo)
{
    //System Exit
    printf("\r\nHandler:Program stop\r\n");     
    DEV_ModuleExit();
	exit(0);
}


uint8_t LCD_Read_Id(void)
{
	uint8_t reg = 0xDC;
	uint8_t tx_val = 0x00;
	uint8_t rx_val;
    LCD_Reset();//Hardware reset
    DEV_Digital_Write(LCD_CS_PIN, 0);
    DEV_Digital_Write(LCD_DC_PIN, 0);
	DEV_HARDWARE_SPI_TransferByte(reg);
	rx_val = DEV_HARDWARE_SPI_TransferByte(tx_val);
    DEV_Digital_Write(LCD_CS_PIN, 1);

	return rx_val;
}

/******************************************************************************
function :	Sends the image buffer in RAM to displays
parameter:
******************************************************************************/
void LCD_Display(UWORD *Image)
{
    UWORD j;

	if(LCD_2_8==id){
		LCD_SetWindow(0,0,LCD_2_8_WIDTH,LCD_2_8_HEIGHT);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_Digital_Write(LCD_DC_PIN, 1);
		LCD_WriteImageBytes((uint8_t *)Image, (uint32_t)LCD_2_8_WIDTH * LCD_2_8_HEIGHT * 2u);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	} else if (LCD_1_8==id) {
		LCD_SetWindow(0,0,LCD_1_8_WIDTH,LCD_1_8_HEIGHT);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_Digital_Write(LCD_DC_PIN, 1);
		LCD_WriteImageBytes((uint8_t *)Image, (uint32_t)LCD_1_8_WIDTH * LCD_1_8_HEIGHT * 2u);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	} else {
		LCD_SetWindow(0,0,LCD_3_5_WIDTH,LCD_3_5_HEIGHT);
		DEV_Digital_Write(LCD_CS_PIN, 0);
		DEV_Digital_Write(LCD_DC_PIN, 1);
		LCD_WriteImageBytes((uint8_t *)Image, (uint32_t)LCD_3_5_WIDTH * LCD_3_5_HEIGHT * 2u);
		DEV_Digital_Write(LCD_CS_PIN, 1);
	}
}

/*******************************************************************************
function:
		Write register data
*******************************************************************************/
static void LCD_Write_AllData(uint16_t Data, uint32_t DataLen)
{
	uint32_t i;
    DEV_Digital_Write(LCD_DC_PIN,1);
    DEV_Digital_Write(LCD_CS_PIN,0);
    for(i = 0; i < DataLen; i++) {
        DEV_SPI_WriteByte(Data >> 8);
        DEV_SPI_WriteByte(Data & 0XFF);
    }
	DEV_Digital_Write(LCD_CS_PIN,1);
}

/********************************************************************************
function:	Set show color
parameter:
		Color  :   Set show color,16-bit depth
********************************************************************************/
void LCD_SetColor(COLOR Color , POINT Xpoint, POINT Ypoint)
{
    LCD_Write_AllData(Color , (uint32_t)Xpoint * (uint32_t)Ypoint);
}

/********************************************************************************
function:	Point (Xpoint, Ypoint) Fill the color
parameter:
	Xpoint :   The x coordinate of the point
	Ypoint :   The y coordinate of the point
	Color  :   Set the color
********************************************************************************/
void LCD_SetPointlColor( POINT Xpoint, POINT Ypoint, COLOR Color)
{
    if ((Xpoint <= sLCD_DIS.LCD_Dis_Column) && (Ypoint <= sLCD_DIS.LCD_Dis_Page)) {
        LCD_SetCursor (Xpoint, Ypoint);
        LCD_SetColor(Color, 1, 1);
    }
}

