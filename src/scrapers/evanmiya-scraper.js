/**
 * EvanMiya Scraper
 * 
 * This script logs into the EvanMiya website and downloads the latest basketball data.
 * It outputs a CSV file to data/em.csv
 */

const { chromium } = require('@playwright/test');
require('dotenv').config();
const path = require('path');
const fs = require('fs');

// Configuration for the scraper
const config = {
  url: 'https://evanmiya.com/',
  outputFile: path.join(__dirname, '../../data/em.csv'),
  maxRetries: 3,
  retryDelay: 2000, // ms
  timeout: 30000, // ms
};

/**
 * Creates directory if it doesn't exist
 * @param {string} dirPath - Path to directory
 */
function ensureDirectoryExists(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`Created directory: ${dirPath}`);
  }
}

/**
 * Sleep function for delays
 * @param {number} ms - Milliseconds to sleep
 * @returns {Promise<void>}
 */
async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Main function to download data from EvanMiya
 * @returns {Promise<Object>} - Result object with success status
 */
async function downloadEvanMiyaData() {
  // Create data directory if it doesn't exist
  const dataDir = path.dirname(config.outputFile);
  ensureDirectoryExists(dataDir);
  
  // Launch browser
  const browser = await chromium.launch();
  const context = await browser.newContext({
    acceptDownloads: true,
    recordVideo: process.env.RECORD_VIDEO === 'true' ? {
      dir: path.join(dataDir, 'videos'),
    } : undefined,
  });

  try {
    console.log('Starting EvanMiya data download process...');
    const page = await context.newPage();
    
    // Go to site with timeout
    await page.goto(config.url, { timeout: config.timeout });
    console.log('Loaded EvanMiya website');
    
    // Accept cookies if prompted
    try {
      await page.getByRole('button', { name: 'OK' }).click({ timeout: 10000 });
      console.log('Accepted cookies');
    } catch (e) {
      console.log('No cookie banner found or already accepted');
    }
    
    // Login to the site
    await page.getByRole('button', { name: 'Login' }).click();
    console.log('Clicked login button');
    
    // Get credentials from environment variables
    const email = process.env.EMAIL;
    const password = process.env.PASSWORD;
    
    if (!email || !password) {
      throw new Error('EMAIL or PASSWORD environment variables not configured');
    }
    
    // Fill login form
    await page.getByRole('textbox', { name: 'Your email' }).fill(email);
    await page.getByRole('textbox', { name: 'Your password' }).fill(password);
    await page.getByRole('button', { name: 'Log in' }).click();
    
    // Wait for login to complete
    await page.waitForLoadState('networkidle', { timeout: config.timeout });
    console.log('Login completed');
    
    // Navigate to Game > Upcoming
    await page.getByRole('link', { name: 'chart-line icon Game' }).click();
    console.log('Navigated to Game section');
    
    await page.getByRole('link', { name: 'angles-right icon Upcoming' }).click();
    console.log('Navigated to Upcoming games');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle', { timeout: config.timeout });
    
    // Navigate to game predictions page
    await page.goto('https://evanmiya.com/?game_predictions', { timeout: config.timeout });
    console.log('Navigated to game predictions');
    
    // Setup download handler and click download button
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'download icon Download' }).click();
    console.log('Clicked download button');
    
    const download = await downloadPromise;
    
    // Save the file to the specified output path
    await download.saveAs(config.outputFile);
    console.log(`File downloaded successfully to: ${config.outputFile}`);
    
    return {
      success: true,
      filePath: config.outputFile,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Error occurred during download process:', error);
    
    // Take screenshot of error state
    try {
      const page = await context.pages()[0];
      if (page) {
        const screenshotPath = path.join(dataDir, `error_${new Date().toISOString().replace(/:/g, '-')}.png`);
        await page.screenshot({ path: screenshotPath });
        console.log(`Error screenshot saved to: ${screenshotPath}`);
      }
    } catch (e) {
      console.error('Failed to capture error screenshot:', e);
    }
    
    return {
      success: false,
      error: error.message,
      timestamp: new Date().toISOString(),
    };
  } finally {
    // Clean up resources
    await context.close();
    await browser.close();
    console.log('Browser closed');
  }
}

/**
 * Run the EvanMiya scraper with retry logic
 * @returns {Promise<Object>} - Result of the scraping operation
 */
async function runWithRetry() {
  let retries = 0;
  let result;
  
  while (retries <= config.maxRetries) {
    if (retries > 0) {
      console.log(`Retry attempt ${retries}/${config.maxRetries}...`);
      await sleep(config.retryDelay);
    }
    
    result = await downloadEvanMiyaData();
    
    if (result.success) {
      break;
    }
    
    retries++;
  }
  
  if (!result.success) {
    console.error(`Failed after ${config.maxRetries} retry attempts`);
  }
  
  return result;
}

// If running this script directly, execute the scraper
if (require.main === module) {
  runWithRetry().then(result => {
    if (result.success) {
      console.log('Download completed successfully');
      process.exit(0);
    } else {
      console.error('Download failed:', result.error);
      process.exit(1);
    }
  });
}

module.exports = { runWithRetry }; 