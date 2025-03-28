import React from 'react';
import type { Language } from '../types';

interface Props {
  language: Language;
  setLanguage: (lang: Language) => void;
}

export const LanguageSelector: React.FC<Props> = ({ language, setLanguage }) => {
  return (
    <div className="inline-block">
      <select
        value={language}
        onChange={(e) => setLanguage(e.target.value as Language)}
        className="px-6 py-2 text-lg rounded-lg border-2 border-green-200 text-green-700 focus:outline-none focus:border-green-500 bg-white cursor-pointer"
      >
        <option value="en">English</option>
        <option value="hi">हिंदी</option>
      </select>
    </div>
  );
};