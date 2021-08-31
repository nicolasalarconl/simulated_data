# %%
#import cupy as cp
import sys
sys.path.append("../../scripts")
from auxiliar.save import save_psf

import cupy as cp
import cv2

from astropy.io import fits
from astropy.utils.data import download_file

DEVICE = int(sys.argv[1])
NAME = str(sys.argv[2]) 
SIZE_FIGURE = int(sys.argv[3]) 
URL = str(sys.argv[4]) 


class Create:
    #TODO : path save a path y size_image size figure
    def __init__(self,device,name_psf,size_figure):
        self.size_figure = size_figure
        self.device = self.init_device(device)
        self.path_gauss = name_psf+'/PSF_'+str(size_figure)+'x'+str(size_figure)+'/PSF_Gauss'
        self.path_real = name_psf+'/PSF_'+str(size_figure)+'x'+str(size_figure)+'/PSF_Real'
        
    def init_device(self,device):
        cp.cuda.Device(device).use()
        return device
    
    def psf_real(self,url):
        image_link = download_file(url, cache=True )
        image = fits.getdata(image_link).astype(cp.float32)
        image = cp.reshape(image,[image.shape[2],image.shape[3]]) 
        image = cv2.resize(cp.asnumpy(image), dsize=(self.size_figure, self.size_figure), interpolation=cv2.INTER_CUBIC)
        psf = cp.array(image)
        type_psf = 'psf_real'
        save_psf(self.path_real,type_psf,cp.asnumpy(psf))
        return psf
     
    def radius(self,psf):  
        value = cp.max(psf)/2
        image = psf > value
        idx,idy = 0,0
        minX = SIZE_FIGURE-1
        maxX = 0
        for row in image:
            for px in row:
                if (px and idx < minX):
                    minX = idx
                if (px and idx > maxX):
                    maxX = idx
                idx = idx+1
            idx = 0
            idy = idy+1
        idx,idy = 0,0
        minY = SIZE_FIGURE-1
        maxY = 0
        for row in image:
            for px in row:
                if (px and idy < minY):
                    minY = idy
                if (px and idy > maxY):
                    maxY = idy
                idx = idx+1
            idx = 0
            idy = idy+1
        Vx=maxX-minX +1
        Vy=maxY-minY +1
        return Vx,Vy
        
    def psf_gauss(self,dx,dy): 
        # parametros
        N=self.size_figure
        xmax=1.0
        pix_size=2.0*xmax/N
        #x=38 # anchos a media altura, son los FWHM
        # =200
        # FWHM=2.355*sigma, sigma=FWHM/2.355
        fwhm_cte=2.0*cp.sqrt(2.0*cp.log(2))
        mux= 0 #50*pix_size # centro corrido 50 pixeles mas alla
        sigmax=dx*pix_size/fwhm_cte # ancho sigma de 2 pixeles
        muy= 0 #100*pix_size # centro corrido 100 pixeles mas alla
        sigmay=dy*pix_size/fwhm_cte # ancho sigma de 3 pixel
        # gaussian en 2-d
        x=cp.linspace(-xmax,xmax,N)
        x=x.reshape((1,N))
        y=x.reshape((N,1))
        gauss=cp.exp(-0.5*((x-mux)/sigmax)**2 -0.5*((y-muy)/sigmay)**2)
        type_psf = 'psf_gauss'
        save_psf(self.path_gauss,type_psf,cp.asnumpy(gauss))
        

    
if __name__ == '__main__':
    print('Create PSF....')
    cr = Create(DEVICE,NAME,SIZE_FIGURE)
    psf_real = cr.psf_real(URL)
    dx,dy = cr.radius(psf_real)
    psf_gauss =  cr.psf_gauss(dx,dy)
    print('Finished')


    