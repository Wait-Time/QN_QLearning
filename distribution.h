#pragma once

#ifndef __DISTRIBUTION_H_
#define __DISTRIBUTION_H_

#include "includes.h"

class distribution
{
private:
    int b;
    int beta;
    std::vector<float> quantiles;
    float tol;
public:
    distribution();
    distribution(int b_para, int beta_para, std::vector<float> quantiles_para)
    {
        this->b = b_para;
        this->beta = beta_para;
        this->quantiles = quantiles_para;
    }
    ~distribution();
    distribution(event_type sampler)
    {
        // Generate sorted samples
        // Apply H&L Method
        /*
        METHOD 10 (“H&L-2”):  The  Pth  percentile value is found by taking that value with  #(np + 0.5).  If this is not an integer, take the interpolated value between    and  # .  As an example, if  
        S5 = (1, 2, 3, 4, 5) and  p = .25, then  #(np + 0.5) = #(1.75) and so  Q1 = 1.75.  
        */
       std::vector<float> samples;
       for (size_t i = 0; i < b; i++)
       {
           samples.push_back(sampler(0));
       }
       
        
    }
    event_type sampler()
    {
        int b_para = this->b;
        std::vector<float> quantiles_para = this->quantiles;
        return [b_para,quantiles_para](float t) -> float { float U = random; return quantiles_para[int(b_para*U)]; };
    }
};

#endif